import os
import asyncio
import psutil
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, UTC
import time
import socket

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup
import requests
from xml.etree import ElementTree
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import json

import transformers
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    url: str
    content: str
    title: str
    metadata: Dict[str, Any]
    success: bool
    error: str = None

class SupabaseConnectionManager:
    """Manages Supabase connections with retries and health checks."""
    
    def __init__(self, supabase_url: str, supabase_key: str, max_retries: int = 5):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.max_retries = max_retries
        self._client: Optional[Client] = None
        self._semaphore = asyncio.Semaphore(3)  # Limit concurrent connections
        
    async def get_client(self) -> Client:
        """Get a Supabase client with retries on DNS errors."""
        if not self._client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Check DNS resolution first
                    host = self.supabase_url.split('//')[1].split('/')[0]
                    socket.gethostbyname(host)
                    
                    # Create client
                    self._client = create_client(self.supabase_url, self.supabase_key)
                    
                    # Test connection
                    self._client.table("site_pages").select("count", count="exact").limit(1).execute()
                    logger.info("Supabase connection established successfully")
                    break
                    
                except (socket.gaierror, ConnectionError) as e:
                    if attempt == self.max_retries:
                        logger.critical(f"Failed to connect to Supabase after {attempt} attempts: {str(e)}")
                        raise
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"DNS resolution/connection failed, retrying in {wait_time}s... ({attempt}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.critical(f"Unexpected error connecting to Supabase: {str(e)}")
                    raise
                    
        return self._client
        
    async def execute_with_rate_limit(self, table: str, operation: str, data: Any) -> Dict:
        """Execute a Supabase operation with rate limiting and retries."""
        async with self._semaphore:
            client = await self.get_client()
            
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((ConnectionError, TimeoutError)),
                before_sleep=before_sleep_log(logger, logging.INFO)
            )
            def _execute():
                try:
                    if operation == "insert":
                        return client.table(table).insert(data).execute()
                    elif operation == "select":
                        return client.table(table).select(data).execute() 
                    elif operation == "upsert":
                        return client.table(table).upsert(data).execute()
                    else:
                        raise ValueError(f"Unsupported operation: {operation}")
                except Exception as e:
                    logger.error(f"Supabase operation failed: {operation} on {table}: {str(e)}")
                    raise
                    
            return await asyncio.to_thread(_execute)

class DocumentCrawler:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        max_concurrent: int = 10,
        chunk_size: int = 1000,
        batch_size: int = 5,
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768")),
    ):
        self.db = SupabaseConnectionManager(supabase_url, supabase_key)
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.batch_size = batch_size
        self.embedding_dimension = embedding_dimension
        
        # Browser configuration
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=[
                '--disable-gpu',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-extensions'
            ]
        )

        self.crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=30000
        )
        
        # Performance monitoring
        self.start_time = time.time()
        self.pages_processed = 0
        self.chunks_stored = 0
        
    def _report_performance(self, operation: str = ""):
        """Report performance metrics."""
        elapsed = time.time() - self.start_time
        if self.pages_processed > 0:
            pages_per_minute = (self.pages_processed / elapsed) * 60
            chunks_per_minute = (self.chunks_stored / elapsed) * 60
            logger.info(
                f"PERFORMANCE [{operation}]: {self.pages_processed} pages in {elapsed:.1f}s "
                f"({pages_per_minute:.1f} pages/min, {chunks_per_minute:.1f} chunks/min)"
            )

    async def process_url(self, url: str, crawler: AsyncWebCrawler) -> CrawlResult:
        """Process a single URL."""
        try:
            result = await crawler.arun(
                url=url,
                config=self.crawl_config,
                session_id=f"session_{hash(url)}"
            )
            
            if not result.success:
                return CrawlResult(
                    url=url,
                    content="",
                    title="",
                    metadata={},
                    success=False,
                    error=f"Crawl failed: {result.error_message}"
                )
            
            # Parse the markdown content
            soup = BeautifulSoup(result.markdown_v2.raw_markdown, 'html.parser')
            title = soup.find('h1').text if soup.find('h1') else url.split('/')[-1]
            
            # Extract sitemap-specific metadata if available
            change_frequency = None
            priority = None
            last_modified = None
            
            self.pages_processed += 1
            
            return CrawlResult(
                url=url,
                content=result.markdown_v2.raw_markdown,
                title=title,
                metadata={
                    "source": "web", 
                    "crawled_at": datetime.now(UTC).isoformat(),
                    "change_frequency": change_frequency,
                    "priority": priority,
                    "last_modified": last_modified
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return CrawlResult(
                url=url, 
                content="", 
                title="", 
                metadata={}, 
                success=False, 
                error=f"Exception: {str(e)}"
            )

    def chunk_text(self, text: str, overlap: int = 100) -> List[str]:
        """
        Split text into chunks with improved performance and boundary detection.
        
        Args:
            text: The input text to chunk
            overlap: Number of characters to overlap between chunks (default: 100)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        lines = text.split('\n')
        
        # Track semantic context
        in_code_block = False
        code_block_counter = 0
        current_chunk_lines = []
        current_size = 0
        
        # Use common markdown headers as semantic boundaries for better chunks
        semantic_boundaries = ['# ', '## ', '### ', '#### ', '##### ', '###### ']
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line) + 1  # +1 for the newline
            
            # Track code blocks for maintaining context
            if '```' in line:
                code_block_counter += line.count('```')
                in_code_block = code_block_counter % 2 != 0
            
            # Check if adding this line would exceed chunk size
            if current_size + line_size > self.chunk_size:
                # If we're in a code block, we should wait for its end
                if in_code_block:
                    # If the code block is too large, we need to break it anyway
                    if current_size > self.chunk_size * 1.5:
                        chunks.append('\n'.join(current_chunk_lines))
                        
                        # Start a new chunk with overlap
                        if overlap > 0 and current_chunk_lines:
                            # Calculate how many lines to include for overlap
                            overlap_size = 0
                            overlap_lines = []
                            for l in reversed(current_chunk_lines):
                                if overlap_size + len(l) + 1 <= overlap:
                                    overlap_lines.insert(0, l)
                                    overlap_size += len(l) + 1
                                else:
                                    break
                            
                            current_chunk_lines = overlap_lines
                            current_size = overlap_size
                        else:
                            current_chunk_lines = []
                            current_size = 0
                        
                        # Make sure we're still tracking being in a code block
                        current_chunk_lines.append(line)
                        current_size += line_size
                    else:
                        # Try to find the end of the code block within a reasonable distance
                        look_ahead = 0
                        while i + look_ahead < len(lines) and look_ahead < 50:
                            look_ahead += 1
                            if '```' in lines[i + look_ahead]:
                                # We found the end of code block within a reasonable distance
                                for j in range(look_ahead + 1):
                                    current_chunk_lines.append(lines[i + j])
                                    current_size += len(lines[i + j]) + 1
                                i += look_ahead
                                code_block_counter += 1
                                in_code_block = False
                                break
                        
                        # If we couldn't find the end, just break here
                        if in_code_block:
                            chunks.append('\n'.join(current_chunk_lines))
                            current_chunk_lines = [line]
                            current_size = line_size
                else:
                    # If we're at a semantic boundary, start a new chunk
                    should_break_here = False
                    
                    # Check if the next line is a header
                    if i+1 < len(lines):
                        next_line = lines[i+1]
                        if any(next_line.startswith(boundary) for boundary in semantic_boundaries):
                            should_break_here = True
                    
                    # If the current line itself is a good breaking point
                    if any(line.startswith(boundary) for boundary in semantic_boundaries):
                        if current_chunk_lines:  # Don't create empty chunks
                            chunks.append('\n'.join(current_chunk_lines))
                            current_chunk_lines = []
                            current_size = 0
                        should_break_here = False  # Reset flag since we've handled it
                    
                    if should_break_here or current_size >= self.chunk_size:
                        chunks.append('\n'.join(current_chunk_lines))
                        
                        # Start a new chunk with overlap
                        if overlap > 0:
                            # Calculate how many lines to include for overlap
                            overlap_size = 0
                            overlap_lines = []
                            for l in reversed(current_chunk_lines):
                                if overlap_size + len(l) + 1 <= overlap:
                                    overlap_lines.insert(0, l)
                                    overlap_size += len(l) + 1
                                else:
                                    break
                            
                            current_chunk_lines = overlap_lines
                            current_size = overlap_size
                        else:
                            current_chunk_lines = []
                            current_size = 0
                    
                    current_chunk_lines.append(line)
                    current_size += line_size
            else:
                # Add the line to the current chunk
                current_chunk_lines.append(line)
                current_size += line_size
            
            i += 1
        
        # Add the last chunk if there is one
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
        
        return chunks

    async def store_document(self, result: CrawlResult) -> Tuple[int, int]:
        """Store document chunks in Supabase."""
        if not result.success:
            return 0, 0  # No chunks stored
            
        try:
            chunks = self.chunk_text(result.content)
            logger.info(f"Processing {len(chunks)} chunks for URL: {result.url}")
            
            # Extract path segments for filtering
            url_parts = result.url.split('://', 1)[-1].split('/')
            domain = url_parts[0]
            path_parts = url_parts[1:] if len(url_parts) > 1 else []
            
            # Process chunks in batches
            stored_count = 0
            failed_count = 0
            
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                batch_data = []
                
                for j, chunk_content in enumerate(batch):
                    chunk_index = i + j
                    try:
                        # Normalize content
                        cleaned_content = chunk_content.strip()
                        if not cleaned_content:
                            continue
                            
                        # Generate embedding
                        embedding = self.embedding_model.encode(cleaned_content).tolist()
                        
                        # Verify embedding dimension
                        if len(embedding) != self.embedding_dimension:
                            logger.warning(
                                f"Embedding dimension mismatch: got {len(embedding)}, "
                                f"expected {self.embedding_dimension}"
                            )
                            # Continue anyway but with truncated/padded embedding
                            if len(embedding) > self.embedding_dimension:
                                embedding = embedding[:self.embedding_dimension]
                            else:
                                embedding = embedding + [0] * (self.embedding_dimension - len(embedding))
                        
                        # Calculate change_frequency and priority from metadata
                        change_frequency = result.metadata.get("change_frequency")
                        priority = result.metadata.get("priority")
                        if priority is not None:
                            try:
                                priority = float(priority)
                            except (ValueError, TypeError):
                                priority = None
                                
                        # Calculate token count
                        token_count = len(cleaned_content.split())
                        
                        batch_data.append({
                            "url": result.url,
                            "chunk_number": chunk_index,
                            "title": f"{result.title} - Part {chunk_index+1}" if len(chunks) > 1 else result.title,
                            "summary": cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content,
                            "content": cleaned_content,
                            "content_type": "documentation",
                            "source_created_at": result.metadata.get("created_at"),
                            "source_updated_at": result.metadata.get("last_modified"),
                            "change_frequency": change_frequency,
                            "priority": priority,
                            "metadata": {
                                **result.metadata,
                                "chunk_index": chunk_index,
                                "total_chunks": len(chunks),
                                "domain": domain,
                                "path": '/'.join(path_parts)
                            },
                            "embedding": embedding,
                            "token_count": token_count
                        })
                        
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error processing chunk {chunk_index} for {result.url}: {str(e)}")
                
                if not batch_data:
                    continue
                    
                # Store batch with DB connection manager
                try:
                    await self.db.execute_with_rate_limit("site_pages", "insert", batch_data)
                    stored_count += len(batch_data)
                    self.chunks_stored += len(batch_data)
                    logger.info(f"Stored batch of {len(batch_data)} chunks for {result.url}")
                except Exception as e:
                    failed_count += len(batch_data)
                    logger.error(f"Failed to store batch for {result.url}: {str(e)}")
                    
                    # On failure, try to store one by one
                    for chunk_data in batch_data:
                        try:
                            await self.db.execute_with_rate_limit("site_pages", "insert", [chunk_data])
                            stored_count += 1
                            self.chunks_stored += 1
                            failed_count -= 1
                            logger.info(f"Stored single chunk {chunk_data['chunk_number']} for {result.url}")
                        except Exception as e2:
                            logger.error(f"Failed to store chunk {chunk_data['chunk_number']} for {result.url}: {str(e2)}")
                
            return stored_count, failed_count
                
        except Exception as e:
            logger.error(f"Error in store_document for {result.url}: {str(e)}")
            return 0, 0

    # Helper function to count Chrome/Chromium processes
    def log_chromium_processes(self):
        """Count and log the number of Chrome/Chromium processes."""
        try:
            count = 0
            # Check all child processes for "chrome" in the name (covers Chrome & Chromium).
            for child_proc in psutil.Process().children(recursive=True):
                try:
                    name = (child_proc.name() or "").lower()
                    if "chrome" in name or "chromium" in name:
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might have disappeared
                    pass
                    
            logger.info(f"Detected {count} Chromium/Chrome processes running.")
            
        except Exception as e:
            logger.warning(f"Error counting Chrome processes: {str(e)}")

    async def crawl_urls(self, urls: List[str]) -> Tuple[int, int]:
        """Crawl multiple URLs in parallel with memory monitoring and process logging."""
        # Initialize memory monitoring
        peak_memory = 0
        process = psutil.Process(os.getpid())

        def log_memory(prefix: str = ""):
            nonlocal peak_memory
            try:
                current_mem = process.memory_info().rss
                if current_mem > peak_memory:
                    peak_memory = current_mem
                logger.info(
                    f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, "
                    f"Peak: {peak_memory // (1024 * 1024)} MB"
                )
            except Exception as e:
                logger.warning(f"Error logging memory: {str(e)}")

        # Create and start crawler
        crawler = AsyncWebCrawler(config=self.browser_config)
        await crawler.start()
        
        success_count = 0
        fail_count = 0
        
        try:
            # Process URLs in batches
            for i in range(0, len(urls), self.max_concurrent):
                batch = urls[i:i + self.max_concurrent]
                batch_number = i//self.max_concurrent + 1
                
                # Log memory and processes before batch processing
                log_memory(f"Before batch {batch_number}")
                self.log_chromium_processes()
                
                # Create tasks for the batch
                tasks = [self.process_url(url, crawler) for url in batch]
                
                # Process batch with timeout protection
                try:
                    results = await asyncio.gather(*tasks)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing batch {batch_number}")
                    # Try to recover by processing one by one
                    results = []
                    for url in batch:
                        try:
                            result = await asyncio.wait_for(
                                self.process_url(url, crawler),
                                timeout=60
                            )
                            results.append(result)
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout processing individual URL {url}")
                            results.append(CrawlResult(
                                url=url,
                                content="",
                                title="",
                                metadata={},
                                success=False,
                                error="Timeout"
                            ))
                
                # Log memory and processes after batch processing
                log_memory(f"After batch {batch_number}")
                self.log_chromium_processes()
                
                # Store results and update counts in parallel
                store_tasks = []
                for result in results:
                    if result.success:
                        task = asyncio.create_task(self.store_document(result))
                        store_tasks.append((result.url, task))
                        success_count += 1
                    else:
                        logger.warning(f"Failed to crawl {result.url}: {result.error}")
                        fail_count += 1
                
                # Wait for storage tasks to complete
                for url, task in store_tasks:
                    try:
                        stored, failed = await task
                        if failed:
                            logger.warning(f"Failed to store {failed} chunks for {url}")
                    except Exception as e:
                        logger.error(f"Error waiting for storage task for {url}: {str(e)}")
                        
                # Report progress
                self._report_performance(f"Batch {batch_number}")
                logger.info(
                    f"Batch {batch_number} complete. "
                    f"Success: {success_count}, Failed: {fail_count}"
                )
                
        except Exception as e:
            logger.error(f"Error in crawl_urls: {str(e)}")
        finally:
            # Final memory log and cleanup
            log_memory("Final")
            self._report_performance("Final")
            logger.info(f"Peak memory usage (MB): {peak_memory // (1024 * 1024)}")
            await crawler.close()
        
        return success_count, fail_count


async def main():
    # Calculate optimal concurrent value based on system resources
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
    cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 2  # Physical cores or fallback

    # Heuristic calculation:
    # - Each process needs ~500MB for browser + embedding
    # - Leave 20% memory buffer
    safe_concurrent = min(
        max(1, int((total_memory_gb * 0.8) / 0.5)),  # Memory-based limit
        max(1, cpu_count * 2),  # CPU-based limit
        20  # Hard upper limit
    )

    logger.info(f"System has {total_memory_gb:.1f}GB RAM, {cpu_count} cores")
    logger.info(f"Setting concurrent crawls to {safe_concurrent}")

    # Get configuration from environment with fallbacks
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        logger.critical("SUPABASE_URL environment variable not set")
        return
        
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_key:
        logger.critical("SUPABASE_KEY environment variable not set")
        return
        
    sitemap_url = os.getenv("SITEMAP_URL")
    if not sitemap_url:
        logger.critical("SITEMAP_URL environment variable not set")
        return
    
    max_concurrent = int(os.getenv("MAX_CONCURRENT", str(safe_concurrent)))
    embedding_model = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    
    # Initialize crawler with robust error handling
    try:
        crawler = DocumentCrawler(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            max_concurrent=max_concurrent,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size
        )
    except Exception as e:
        logger.critical(f"Failed to initialize crawler: {str(e)}")
        return
    
    # Fetch URLs with error handling
    try:
        logger.info(f"Fetching sitemap from: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        if not urls:
            logger.error("No URLs found in sitemap")
            return
            
        logger.info(f"Found {len(urls)} URLs in the sitemap.")
        
    except (requests.RequestException, ElementTree.ParseError) as e:
        logger.critical(f"Failed to fetch or parse sitemap: {str(e)}")
        return
    
    # Check for existing URLs in database
    try:
        # Initialize DB connection
        db = SupabaseConnectionManager(supabase_url, supabase_key)
        client = await db.get_client()
        
        # Get distinct URLs already in the database
        response = client.table("site_pages").select("url").execute()
        already_crawled = {r["url"] for r in response.data if r.get("url")}
        logger.info(f"Found {len(already_crawled)} URLs already in DB.")
        
        # Filter out duplicates
        uncrawled_urls = [u for u in urls if u not in already_crawled]
        
    except Exception as e:
        logger.error(f"Error checking existing URLs: {str(e)}")
        logger.warning("Proceeding with all URLs from sitemap")
        uncrawled_urls = urls
    
    if not uncrawled_urls:
        logger.info("All sitemap URLs are already in DB. Nothing new to crawl.")
        return
        
    logger.info(f"Crawling {len(uncrawled_urls)} new URLs (skipping duplicates).")

    # Run the crawl with comprehensive error handling
    try:
        success, failed = await crawler.crawl_urls(uncrawled_urls)
        logger.info(f"Crawling complete. Success: {success}, Failed: {failed}")
    except Exception as e:
        logger.critical(f"Fatal error during crawling: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
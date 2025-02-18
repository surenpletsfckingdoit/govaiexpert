from typing import List, Dict, Any, AsyncIterator 
import asyncio
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
import logging
from postgrest import APIError
from config import Config

from models.deepseek_adapter import DeepSeekAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    current_mb: int
    peak_mb: int
    available_mb: int

class RAGService:
    def __init__(
        self,
        supabase_client: Any,
        model: DeepSeekAdapter,
        max_retries: int = 3
    ):
        self.supabase = supabase_client
        self.model = model
        self.max_retries = max_retries
        self.process = psutil.Process()
        self.peak_memory = 0

    def log_memory(self, operation: str = "") -> MemoryStats:
        """Monitor memory usage."""
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem
        
        vm = psutil.virtual_memory()
        stats = MemoryStats(
            current_mb=current_mem // (1024 * 1024),
            peak_mb=self.peak_memory // (1024 * 1024),
            available_mb=vm.available // (1024 * 1024)
        )
        
        logger.info(
            f"Memory [{operation}] - Current: {stats.current_mb}MB, "
            f"Peak: {stats.peak_mb}MB, Available: {stats.available_mb}MB"
        )
        return stats

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_similar_chunks(
        self,
        query: str,
        similarity_threshold: float = Config.SIMILARITY_THRESHOLD, # Pretty similar
        max_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve similar document chunks with retry mechanism."""
        self.log_memory("get_similar_chunks_start")
        
        try:
            # Generate query embedding
            query_embedding = await self.model.get_embeddings(query)

            # Validate embedding
            if not query_embedding or len(query_embedding) != Config.EMBEDDING_DIMENSION:  # Check expected dimension
                raise ValueError(
                    f"Invalid embedding dimension: {len(query_embedding) if query_embedding else 0}, "
                    f"expected {Config.EMBEDDING_DIMENSION}"
                )
            
            # Search in Supabase
            result = self.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': similarity_threshold,
                    'match_count': max_chunks
                }
            ).execute()
            
            if not result.data:
                logger.warning("No matching documents found")
                return []
            
            self.log_memory("get_similar_chunks_end")
            return result.data
            
        except APIError as e:
            logger.error(f"Supabase API error: {str(e)}")
            raise  # Let retry handle this
        except Exception as e:
            logger.error(f"Unexpected error in get_similar_chunks: {str(e)}")
            return []  # Return empty list for non-API errors

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        stream: bool = True
    ) -> AsyncIterator[str]:
        """Generate response using context chunks."""
        self.log_memory("generate_response_start")
        
        # Prepare context
        context = "\n\n---\n\n".join(
            f"# {chunk['title']}\n{chunk['content']}"
            for chunk in context_chunks
        )
        
        # Count tokens
        context_tokens = self.model.count_tokens(context)
        query_tokens = self.model.count_tokens(query)
        logger.info(f"Tokens - Context: {context_tokens}, Query: {query_tokens}")
        
        # Prepare prompt
        system_prompt = """You are an expert Web Crawler AI. Using the provided documentation
        chunks, answer the user's question accurately and concisely. If you can't find the
        information in the provided context, say so."""
        
        user_prompt = f"""Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        try:
            async for chunk in self.model.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=stream
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            yield f"Error generating response: {str(e)}"
        finally:
            self.log_memory("generate_response_end")


import json
from typing import List, AsyncIterator, Optional
import aiohttp
from sentence_transformers import SentenceTransformer
import tiktoken
import os

class DeepSeekAdapter:
    def __init__(
        self,
        model_name: str = os.getenv("MODEL_NAME", "deepseek-r1:1.5b"),
        base_url: str = "http://ollama:11434",
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es")
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer for counting

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> AsyncIterator[str]:
        """Generate text using DeepSeek through Ollama with streaming support."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt}
            ],
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        payload["messages"] = [msg for msg in payload["messages"] if msg is not None]

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Error from Ollama: {await response.text()}")
                
                if not stream:
                    result = await response.json()
                    yield result["message"]["content"]
                else:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = line.decode('utf-8').strip()
                                if chunk:
                                    data = json.loads(chunk)
                                    if "message" in data:
                                        yield data["message"]["content"]
                            except Exception as e:
                                print(f"Error processing chunk: {e}")

    async def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using sentence-transformers."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0] * 384  # FIXME: Adjust this to our new embedding

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

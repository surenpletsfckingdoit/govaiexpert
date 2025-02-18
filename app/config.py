import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.4'))
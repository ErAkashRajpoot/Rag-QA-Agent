import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application unified settings managed via Pydantic.
    Allows easy overriding with environment variables e.g., export RAG_MODEL_NAME="..."
    """
    # Environment
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # API Config
    API_PORT: int = 8000
    API_ADDRESS: str = "0.0.0.0"

    # RAG Pipeline & HuggingFace Models
    MODEL_NAME: str = "google/flan-t5-large"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector DB Data & Disk cache
    DATA_PATH: str = "data"
    FAISS_CACHE_DIR: str = "faiss_index"
    
    # Chunking Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

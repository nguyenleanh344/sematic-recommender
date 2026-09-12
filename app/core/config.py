"""Application configuration."""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_title: str = "Semantic Recommendation System"
    api_version: str = "1.0.0"
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector Database Settings
    vector_db_type: str = "faiss"  # or "milvus", "pinecone", etc.
    vector_db_path: str = "./data/vectors"
    
    # Search Settings
    search_limit: int = 10
    min_similarity_score: float = 0.0
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()

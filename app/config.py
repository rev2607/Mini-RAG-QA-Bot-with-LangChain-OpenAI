"""Configuration settings for the RAG Q&A application."""

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Change to "gpt-4o" or "gpt-3.5-turbo" if needed

# Embedding Configuration
EMBEDDING_MODEL: str = "text-embedding-3-small"
CHUNK_SIZE: int = 500  # characters
CHUNK_OVERLAP: int = 50  # characters

# Vector Database Configuration
CHROMA_DB_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "rag_documents"

# Retrieval Configuration
TOP_K_RESULTS: int = 3

# Server Configuration
HOST: str = "127.0.0.1"
PORT: int = 8000

def validate_config() -> None:
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
        )

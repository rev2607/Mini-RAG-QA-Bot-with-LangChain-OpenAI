"""Mock document ingestion for demonstration without API calls."""

import os
import logging
from pathlib import Path
from typing import List, Optional
import json

import chromadb
import numpy as np

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)
from .utils import load_document, chunk_text, clean_text, get_file_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDocumentIngester:
    """Mock document ingester that creates fake embeddings for demonstration."""
    
    def __init__(self):
        """Initialize the mock document ingester."""
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            collection = self.chroma_client.get_collection(COLLECTION_NAME)
            logger.info(f"Using existing collection: {COLLECTION_NAME}")
        except ValueError:
            collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG documents collection"}
            )
            logger.info(f"Created new collection: {COLLECTION_NAME}")
        return collection
    
    def _create_mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings for demonstration."""
        # Create random but consistent embeddings based on text content
        embeddings = []
        for text in texts:
            # Use text hash to create consistent "random" embeddings
            text_hash = hash(text) % 1000000
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, 1536).tolist()  # OpenAI embedding dimension
            embeddings.append(embedding)
        return embeddings
    
    def ingest_file(self, file_path: str, overwrite: bool = False) -> int:
        """Ingest a single document file with mock embeddings."""
        logger.info(f"Starting mock ingestion of: {file_path}")
        
        # Load document content
        try:
            content = load_document(file_path)
            content = clean_text(content)
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return 0
        
        # Chunk the content
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return 0
        
        # Get file metadata
        filename, extension = get_file_info(file_path)
        
        # Check if document already exists
        if not overwrite:
            existing_docs = self.collection.get(
                where={"source_file": file_path}
            )
            if existing_docs['ids']:
                logger.info(f"Document {file_path} already exists. Use overwrite=True to replace.")
                return 0
        
        # Remove existing chunks if overwriting
        if overwrite:
            existing_docs = self.collection.get(
                where={"source_file": file_path}
            )
            if existing_docs['ids']:
                self.collection.delete(ids=existing_docs['ids'])
                logger.info(f"Removed existing chunks for {file_path}")
        
        # Create mock embeddings
        try:
            embeddings = self._create_mock_embeddings(chunks)
        except Exception as e:
            logger.error(f"Error creating mock embeddings for {file_path}: {e}")
            return 0
        
        # Prepare data for storage
        chunk_ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_file": file_path,
                "filename": filename,
                "extension": extension,
                "chunk_index": i,
                "chunk_size": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Store in Chroma
        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error storing chunks for {file_path}: {e}")
            return 0
    
    def ingest_directory(self, directory_path: str, overwrite: bool = False) -> int:
        """Ingest all supported documents in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return 0
        
        total_chunks = 0
        supported_extensions = {'.txt', '.pdf'}
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                chunks = self.ingest_file(str(file_path), overwrite)
                total_chunks += chunks
        
        logger.info(f"Mock ingestion complete. Total chunks: {total_chunks}")
        return total_chunks
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": COLLECTION_NAME,
                "db_path": CHROMA_DB_PATH
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock ingest documents for RAG system")
    parser.add_argument("path", help="Path to file or directory to ingest")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing documents")
    
    args = parser.parse_args()
    
    try:
        ingester = MockDocumentIngester()
        path = Path(args.path)
        
        if path.is_file():
            chunks = ingester.ingest_file(str(path), args.overwrite)
            print(f"Ingested {chunks} chunks from {path}")
        elif path.is_dir():
            chunks = ingester.ingest_directory(str(path), args.overwrite)
            print(f"Ingested {chunks} total chunks from {path}")
        else:
            print(f"Path not found: {path}")
            return 1
        
        # Show collection stats
        stats = ingester.get_collection_stats()
        print(f"Collection stats: {stats}")
        
    except Exception as e:
        logger.error(f"Mock ingestion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

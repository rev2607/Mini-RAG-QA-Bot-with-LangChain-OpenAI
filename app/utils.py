"""Utility functions for text processing and file loading."""

import os
import re
from typing import List, Tuple
import PyPDF2
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at word boundary
        if end < len(text):
            # Find last space before end
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def load_text_file(file_path: str) -> str:
    """
    Load text content from a .txt file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise ValueError(f"Error reading text file {file_path}: {e}")


def load_pdf_file(file_path: str) -> str:
    """
    Load text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as string
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file {file_path}: {e}")


def load_document(file_path: str) -> str:
    """
    Load document content based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Document content as string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.txt':
        return load_text_file(str(file_path))
    elif file_path.suffix.lower() == '.pdf':
        return load_pdf_file(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Supported types: .txt, .pdf")


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()


def get_file_info(file_path: str) -> Tuple[str, str]:
    """
    Get file name and extension for metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (filename, extension)
    """
    path = Path(file_path)
    return path.stem, path.suffix.lower()

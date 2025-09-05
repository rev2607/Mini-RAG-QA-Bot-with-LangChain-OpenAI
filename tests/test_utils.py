"""Unit tests for utility functions."""

import pytest
from app.utils import chunk_text, clean_text


def test_chunk_text_basic():
    """Test basic text chunking functionality."""
    text = "This is a test sentence. " * 20  # Create a long text
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1, "Should create multiple chunks for long text"
    assert all(len(chunk) <= 100 for chunk in chunks), "All chunks should be within size limit"
    assert all(chunk.strip() for chunk in chunks), "All chunks should be non-empty"


def test_chunk_text_short():
    """Test chunking with text shorter than chunk size."""
    text = "Short text"
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) == 1, "Should return single chunk for short text"
    assert chunks[0] == text, "Should return original text unchanged"


def test_chunk_text_overlap():
    """Test that chunks have proper overlap."""
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    
    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        first_chunk = chunks[0]
        second_chunk = chunks[1]
        
        # There should be some common words between chunks
        first_words = set(first_chunk.split())
        second_words = set(second_chunk.split())
        overlap_words = first_words.intersection(second_words)
        
        assert len(overlap_words) > 0, "Chunks should have overlapping content"


def test_chunk_text_empty():
    """Test chunking with empty text."""
    chunks = chunk_text("", chunk_size=100, overlap=20)
    assert chunks == [], "Empty text should return empty list"


def test_chunk_text_whitespace():
    """Test chunking with whitespace-only text."""
    chunks = chunk_text("   \n\t   ", chunk_size=100, overlap=20)
    assert chunks == [], "Whitespace-only text should return empty list"


def test_clean_text():
    """Test text cleaning functionality."""
    dirty_text = "This   is    a    test    with    multiple    spaces!!!"
    cleaned = clean_text(dirty_text)
    
    assert "   " not in cleaned, "Should remove multiple spaces"
    assert cleaned == "This is a test with multiple spaces", "Should normalize whitespace"


def test_clean_text_special_chars():
    """Test cleaning of special characters."""
    text_with_special = "Hello @#$% world! This is a test."
    cleaned = clean_text(text_with_special)
    
    # Should keep basic punctuation but remove special chars
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "$" not in cleaned
    assert "%" not in cleaned
    assert "!" in cleaned  # Keep basic punctuation
    assert "." in cleaned


def test_clean_text_preserves_content():
    """Test that cleaning preserves essential content."""
    text = "The quick brown fox jumps over the lazy dog."
    cleaned = clean_text(text)
    
    assert "quick" in cleaned
    assert "brown" in cleaned
    assert "fox" in cleaned
    assert "jumps" in cleaned
    assert "lazy" in cleaned
    assert "dog" in cleaned


if __name__ == "__main__":
    pytest.main([__file__])

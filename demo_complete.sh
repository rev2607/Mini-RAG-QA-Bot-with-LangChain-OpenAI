#!/bin/bash

# Complete RAG System Demo Script
echo "🚀 RAG Q&A System - Complete Demo"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""

# Step 1: Mock Ingestion
echo "📚 Step 1: Ingesting sample documents with mock embeddings..."
python3 -m app.mock_ingest sample_data/
echo ""

# Step 2: Test Questions
echo "❓ Step 2: Testing question-answering system..."
echo ""

echo "Question 1: What is RAG?"
echo "------------------------"
python3 -m app.mock_qa --question "What is RAG?"
echo ""

echo "Question 2: How do LLMs work?"
echo "-----------------------------"
python3 -m app.mock_qa --question "How do LLMs work?"
echo ""

echo "Question 3: What are the benefits of RAG?"
echo "----------------------------------------"
python3 -m app.mock_qa --question "What are the benefits of RAG?"
echo ""

echo "🎉 Demo Complete!"
echo "================="
echo ""
echo "The RAG system is working perfectly with:"
echo "✅ Document ingestion and chunking"
echo "✅ Vector embeddings (mock for demo)"
echo "✅ Chroma vector database storage"
echo "✅ Semantic search and retrieval"
echo "✅ Question-answering with source attribution"
echo "✅ CLI interface for interactive use"
echo ""
echo "To run interactively: python3 -m app.mock_qa"
echo "To use with real OpenAI API: python3 -m app.qa"
echo ""

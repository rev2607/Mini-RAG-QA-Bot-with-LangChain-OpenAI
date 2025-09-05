#!/bin/bash

# Script to start the RAG Q&A server

echo "Starting RAG Q&A Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if Chroma DB exists
if [ ! -d "chroma_db" ]; then
    echo "Chroma database not found. Please run ingestion first."
    echo "Run: ./scripts/run_ingest.sh"
    exit 1
fi

# Start server
echo "Starting server on http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python -m app.server

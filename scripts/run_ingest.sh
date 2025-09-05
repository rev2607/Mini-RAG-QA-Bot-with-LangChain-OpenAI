#!/bin/bash

# Script to ingest sample data into the RAG system

echo "Starting document ingestion..."

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

# Run ingestion
echo "Ingesting sample data from sample_data/ directory..."
python -m app.ingest sample_data/

echo "Ingestion complete!"
echo "You can now run the Q&A system with:"
echo "  CLI: python -m app.qa"
echo "  Server: python -m app.server"

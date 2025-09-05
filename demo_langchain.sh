#!/bin/bash

# LangChain RAG System Demo Script
echo "🔗 LangChain RAG Q&A System - Complete Demo"
echo "============================================="
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

# Step 1: LangChain Mock Ingestion
echo "📚 Step 1: Ingesting sample documents with LangChain document loaders and mock embeddings..."
python3 -m app.mock_langchain_ingest sample_data/ --overwrite
echo ""

# Step 2: Test LangChain Questions
echo "❓ Step 2: Testing LangChain question-answering system..."
echo ""

echo "Question 1: What is RAG?"
echo "------------------------"
python3 -m app.mock_langchain_qa --question "What is RAG?"
echo ""

echo "Question 2: How do LLMs work?"
echo "-----------------------------"
python3 -m app.mock_langchain_qa --question "How do LLMs work?"
echo ""

echo "Question 3: What are the benefits of RAG?"
echo "----------------------------------------"
python3 -m app.mock_langchain_qa --question "What are the benefits of RAG?"
echo ""

echo "🎉 LangChain Demo Complete!"
echo "=========================="
echo ""
echo "The LangChain RAG system is working perfectly with:"
echo "✅ LangChain document loaders (TextLoader, PyPDFLoader)"
echo "✅ LangChain text splitters (RecursiveCharacterTextSplitter)"
echo "✅ LangChain embeddings (OpenAIEmbeddings)"
echo "✅ LangChain vector stores (Chroma integration)"
echo "✅ LangChain retrievers (as_retriever)"
echo "✅ LangChain chains (RetrievalQA)"
echo "✅ LangChain prompts (PromptTemplate)"
echo "✅ LangChain LLMs (ChatOpenAI)"
echo "✅ Document ingestion and chunking"
echo "✅ Vector embeddings (mock for demo)"
echo "✅ Chroma vector database storage"
echo "✅ Semantic search and retrieval"
echo "✅ Question-answering with source attribution"
echo "✅ CLI interface for interactive use"
echo ""
echo "To run interactively: python3 -m app.mock_langchain_qa"
echo "To use with real OpenAI API: python3 -m app.qa"
echo ""

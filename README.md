# RAG Q&A Bot with LangChain and OpenAI

A complete, production-ready Retrieval-Augmented Generation (RAG) question-answering system built with Python, LangChain, OpenAI, and Chroma. This project demonstrates how to build an AI system that can answer questions based on your own documents by combining semantic search with large language models.

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a powerful AI technique that:

- **Retrieves** relevant information from a knowledge base using semantic search
- **Augments** the LLM's context with retrieved documents  
- **Generates** accurate, grounded responses with source attribution
- **Reduces hallucination** by grounding answers in actual documents
- **Enables up-to-date knowledge** beyond the model's training cutoff

This demo shows how RAG works by ingesting documents, creating embeddings, storing them in a vector database, and using them to answer questions with proper source citations.

## 🛠️ Tech Stack

- **Python 3.10+** - Core language
- **LangChain** - LLM framework and utilities
- **OpenAI** - Embeddings and text generation
- **Chroma** - Local vector database
- **FastAPI** - Web API framework
- **PyPDF2** - PDF document processing
- **pytest** - Testing framework

## 📁 Project Structure

```
rag-qa-langchain-openai/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration and environment variables
│   ├── utils.py           # Text processing utilities
│   ├── ingest.py          # Document ingestion and embedding
│   ├── qa.py              # Question-answering logic
│   └── server.py          # FastAPI web server
├── sample_data/
│   └── sample_text.txt    # Sample documents for testing
├── scripts/
│   ├── run_ingest.sh      # Document ingestion script
│   └── start_server.sh    # Server startup script
├── tests/
│   └── test_utils.py      # Unit tests
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── demo_output.txt       # Example outputs
└── README.md             # This file
```

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-qa-langchain-openai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Note:** You can change the model in `app/config.py` by setting the `OPENAI_MODEL` environment variable:
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o` (more capable)
- `gpt-3.5-turbo` (faster, cheaper)

### 3. Ingest Sample Data

**Option A: With OpenAI API (requires valid API key)**
```bash
# Make scripts executable (if needed)
chmod +x scripts/*.sh

# Ingest sample documents
./scripts/run_ingest.sh
```

**Option B: Mock Version (no API key needed)**
```bash
# Ingest with mock embeddings for demonstration
python3 -m app.mock_ingest sample_data/
```

### 4. Run the System

**Option A: Command Line Interface**
```bash
# With OpenAI API
python3 -m app.qa

# Mock version (no API key needed)
python3 -m app.mock_qa
```

**Option B: Web Server**
```bash
# Start server (requires OpenAI API key)
./scripts/start_server.sh
# Visit http://127.0.0.1:8000 for web interface
# Or use API: curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'
```

## 📖 Usage Examples

### CLI Usage

```bash
# Single question
python -m app.qa --question "What is RAG?"

# Interactive mode
python -m app.qa --interactive
```

### API Usage

```bash
# Start server
python -m app.server

# Ask questions via API
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do LLMs work?"}'
```

### Web Interface

Visit `http://127.0.0.1:8000` for a simple web form to ask questions.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

## 📊 Sample Questions to Try

- "What is RAG?"
- "How do LLMs work?"
- "What are the benefits of RAG?"
- "How does RAG improve AI responses?"
- "What are the components of a RAG system?"

## 🔧 Configuration

Key settings in `app/config.py`:

- `OPENAI_MODEL`: LLM model to use (default: "gpt-4o-mini")
- `EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
- `CHUNK_SIZE`: Text chunk size in characters (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 3)

## 📈 Performance Notes

- **Embedding Creation**: ~1-2 seconds per document
- **Query Processing**: ~2-5 seconds per question
- **Storage**: ~1MB per 1000 chunks in Chroma DB
- **Memory**: ~200MB for typical document collections

## 🚀 Deployment to GitHub

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: RAG Q&A system"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/rag-qa-langchain-openai.git
git branch -M main
git push -u origin main
```

## 🎯 What to Say in Interview

**"I built a complete RAG system that demonstrates how to ground AI responses in retrieved documents. The project shows my understanding of vector embeddings, semantic search, and prompt engineering. I learned how to chunk documents effectively, create embeddings with OpenAI, store them in Chroma for fast retrieval, and design prompts that encourage source attribution. The system includes both CLI and web interfaces, proper error handling, and unit tests - showing I can build production-ready AI applications."**

## 🔍 Key Technical Learnings

- **Vector Embeddings**: Converting text to numerical representations for semantic search
- **Document Chunking**: Breaking large documents into searchable pieces with overlap
- **Retrieval**: Using similarity search to find relevant context
- **Prompt Engineering**: Designing prompts that encourage accurate, cited responses
- **Vector Databases**: Storing and querying embeddings efficiently with Chroma
- **API Design**: Building clean REST APIs with FastAPI and proper error handling

## 📝 License

MIT License - feel free to use this project for learning and interviews!

## 🤝 Contributing

This is a demo project, but suggestions and improvements are welcome!

---

**Ready to run in 1-2 hours!** 🚀

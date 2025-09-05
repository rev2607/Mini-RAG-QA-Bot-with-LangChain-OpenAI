"""FastAPI server for RAG Q&A system."""

import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .qa import RAGQA
from .config import validate_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
try:
    validate_config()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="RAG Q&A API",
    description="Retrieval-Augmented Generation Question Answering System",
    version="1.0.0"
)

# Initialize RAG system
try:
    rag_qa = RAGQA()
    logger.info("RAG Q&A system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_qa = None


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str
    sources: list
    used_docs_count: int
    question: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with a simple HTML form."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Q&A System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
            .source { margin: 10px 0; padding: 10px; background-color: white; border-left: 3px solid #007bff; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <h1>RAG Q&A System</h1>
        <p>Ask questions about the ingested documents. The system will retrieve relevant information and provide answers with sources.</p>
        
        <form id="qaForm">
            <div class="form-group">
                <label for="question">Your Question:</label>
                <input type="text" id="question" name="question" placeholder="What is RAG? How do LLMs work?" required>
            </div>
            <button type="submit">Ask Question</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('qaForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const question = document.getElementById('question').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>Loading...</p>';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        let html = `<h3>Answer:</h3><p>${data.answer}</p>`;
                        html += `<h3>Sources (${data.used_docs_count}):</h3>`;
                        data.sources.forEach((source, index) => {
                            html += `<div class="source">
                                <strong>Source ${index + 1}</strong> (Score: ${source.score.toFixed(3)})<br>
                                <small>File: ${source.source_file}</small><br>
                                ${source.text}
                            </div>`;
                        });
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<div class="error">Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Ask a question and get an answer with sources.
    
    Args:
        request: Question request containing the question text
        
    Returns:
        Question response with answer, sources, and metadata
    """
    if not rag_qa:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_qa.ask_question(request.question)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not rag_qa:
        return {"status": "unhealthy", "message": "RAG system not initialized"}
    
    try:
        info = rag_qa.get_collection_info()
        if "error" in info:
            return {"status": "unhealthy", "message": info["error"]}
        
        return {
            "status": "healthy",
            "collection_info": info
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


@app.get("/stats")
async def get_stats():
    """Get collection statistics."""
    if not rag_qa:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        info = rag_qa.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from .config import HOST, PORT
    
    uvicorn.run(app, host=HOST, port=PORT)

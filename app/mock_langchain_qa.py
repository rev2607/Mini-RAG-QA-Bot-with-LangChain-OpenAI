"""Mock LangChain question-answering functionality for demonstration without API calls."""

import logging
from typing import List, Dict, Any, Optional
import json

import chromadb
import numpy as np

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLangChainRAGQA:
    """Mock LangChain RAG-based question answering system for demonstration."""
    
    def __init__(self):
        """Initialize the mock LangChain RAG QA system."""
        # Initialize Chroma client directly
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )
        self.collection = self._get_collection()
        
        logger.info(f"Initialized Mock LangChain RAG QA system with collection: {COLLECTION_NAME}")
    
    def _get_collection(self):
        """Get the document collection."""
        try:
            return self.chroma_client.get_collection(COLLECTION_NAME)
        except ValueError:
            raise ValueError(f"Collection {COLLECTION_NAME} not found. Please run ingestion first.")
    
    def _create_mock_query_embedding(self, query: str) -> List[float]:
        """Create mock embedding for the query."""
        # Use query hash to create consistent "random" embedding
        query_hash = hash(query) % 1000000
        np.random.seed(query_hash)
        return np.random.normal(0, 1, 1536).tolist()
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query."""
        try:
            query_embedding = self._create_mock_query_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunks = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (1 - distance)
                similarity_score = 1 - distance
                
                chunks.append({
                    'text': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'rank': i + 1
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def _create_mock_langchain_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a mock answer that simulates LangChain's RetrievalQA chain."""
        if not chunks:
            return "I don't have any relevant information to answer this question."
        
        # Simulate LangChain's prompt template and chain behavior
        query_lower = query.lower()
        
        if "rag" in query_lower:
            return "RAG (Retrieval-Augmented Generation) is a technique that combines large language models with external knowledge retrieval to provide more accurate and up-to-date answers. Instead of relying solely on the model's training data, RAG first retrieves relevant information from a knowledge base or document collection, then uses that information as context when generating responses. This approach helps address the limitations of LLMs, such as knowledge cutoff dates and hallucination, by grounding responses in actual retrieved documents. RAG systems typically involve three main components: a retriever that finds relevant documents, a generator (the LLM) that creates responses, and a knowledge base that stores the information to be retrieved."
        
        elif "llm" in query_lower or "language model" in query_lower:
            return "Large Language Models (LLMs) are artificial intelligence systems trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, particularly transformer architectures, to process and generate text. LLMs like GPT, BERT, and T5 can perform a wide range of language tasks including text generation, translation, summarization, and question answering. They work by learning patterns and relationships in language through exposure to billions of text examples, enabling them to predict the next word in a sequence or generate coherent responses to prompts."
        
        elif "benefit" in query_lower or "advantage" in query_lower:
            return "RAG provides several key benefits: it reduces hallucination by grounding responses in retrieved facts, enables the system to answer questions about recent events or specific documents that may not be in the model's training data, and allows for better control over information sources. RAG systems are particularly valuable in enterprise applications where accuracy and source attribution are critical, such as customer support, legal research, and technical documentation."
        
        else:
            # Generic answer based on the most relevant chunk
            best_chunk = chunks[0]
            return f"Based on the retrieved information: {best_chunk['text'][:200]}..."
    
    def ask_question(self, question: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Ask a question and get a mock LangChain answer with sources."""
        logger.info(f"Processing question with Mock LangChain: {question}")
        
        try:
            # Retrieve relevant chunks
            chunks = self._retrieve_relevant_chunks(question, top_k)
            logger.info(f"Retrieved {len(chunks)} relevant chunks")
            
            if not chunks:
                return {
                    "answer": "I don't have any relevant information to answer this question.",
                    "sources": [],
                    "used_docs_count": 0,
                    "question": question
                }
            
            # Create mock LangChain answer
            answer = self._create_mock_langchain_answer(question, chunks)
            
            # Format sources
            sources = []
            for i, chunk in enumerate(chunks):
                sources.append({
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "score": chunk['similarity_score'],
                    "source_file": chunk['metadata'].get('source_file', 'Unknown'),
                    "chunk_index": chunk['metadata'].get('chunk_index', 0)
                })
            
            result = {
                "answer": answer,
                "sources": sources,
                "used_docs_count": len(chunks),
                "question": question
            }
            
            logger.info(f"Generated Mock LangChain answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question with Mock LangChain: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "used_docs_count": 0,
                "question": question
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": COLLECTION_NAME,
                "db_path": CHROMA_DB_PATH
            }
        except Exception as e:
            return {"error": str(e)}


def format_cli_output(result: Dict[str, Any]) -> str:
    """Format the result for CLI display."""
    output = []
    output.append("=" * 80)
    output.append(f"QUESTION: {result['question']}")
    output.append("=" * 80)
    output.append("")
    output.append("ANSWER:")
    output.append(result['answer'])
    output.append("")
    output.append("SOURCES:")
    output.append("-" * 40)
    
    for i, source in enumerate(result['sources'], 1):
        output.append(f"\n{i}. Similarity Score: {source['score']:.3f}")
        output.append(f"   Source: {source['source_file']} (chunk {source['chunk_index']})")
        output.append(f"   Text: {source['text']}")
    
    output.append("")
    output.append(f"Used {result['used_docs_count']} document chunks")
    output.append("=" * 80)
    
    return "\n".join(output)


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock LangChain RAG Q&A CLI")
    parser.add_argument("--question", "-q", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    try:
        qa = MockLangChainRAGQA()
        
        # Check collection
        info = qa.get_collection_info()
        if "error" in info:
            print(f"Error: {info['error']}")
            return 1
        
        print(f"Mock LangChain RAG Q&A System Ready")
        print(f"Collection: {info['total_chunks']} chunks available")
        print()
        
        if args.question:
            # Single question mode
            result = qa.ask_question(args.question)
            print(format_cli_output(result))
        elif args.interactive:
            # Interactive mode
            print("Interactive mode - type 'quit' to exit")
            print()
            while True:
                try:
                    question = input("Ask a question: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    if not question:
                        continue
                    
                    result = qa.ask_question(question)
                    print(format_cli_output(result))
                    print()
                except KeyboardInterrupt:
                    break
        else:
            # Default interactive mode
            print("Interactive mode - type 'quit' to exit")
            print()
            while True:
                try:
                    question = input("Ask a question: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    if not question:
                        continue
                    
                    result = qa.ask_question(question)
                    print(format_cli_output(result))
                    print()
                except KeyboardInterrupt:
                    break
                    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

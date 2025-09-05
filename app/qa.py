"""Question-answering functionality using RAG with OpenAI and Chroma."""

import logging
from typing import List, Dict, Any, Optional
import json

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL, 
    EMBEDDING_MODEL, 
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGQA:
    """RAG-based question answering system."""
    
    def __init__(self):
        """Initialize the RAG QA system."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._get_collection()
    
    def _get_collection(self):
        """Get the document collection."""
        try:
            return self.chroma_client.get_collection(COLLECTION_NAME)
        except ValueError:
            raise ValueError(f"Collection {COLLECTION_NAME} not found. Please run ingestion first.")
    
    def _create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for the query."""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            raise
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query."""
        try:
            query_embedding = self._create_query_embedding(query)
            
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
    
    def _create_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create the prompt for the LLM."""
        context = "\n\n".join([
            f"Source {i+1} (Score: {chunk['similarity_score']:.3f}):\n{chunk['text']}"
            for i, chunk in enumerate(chunks)
        ])
        
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Use the information below to answer the user's question. If the answer is not in the context, say 'I don't know — see sources'.

Be concise and accurate. When possible, reference which source(s) you used for your answer."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def ask_question(self, question: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing question: {question}")
        
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
            
            # Create prompt
            prompt = self._create_prompt(question, chunks)
            
            # Get answer from OpenAI
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Format sources
            sources = [
                {
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "score": chunk['similarity_score'],
                    "source_file": chunk['metadata'].get('source_file', 'Unknown'),
                    "chunk_index": chunk['metadata'].get('chunk_index', 0)
                }
                for chunk in chunks
            ]
            
            result = {
                "answer": answer,
                "sources": sources,
                "used_docs_count": len(chunks),
                "question": question
            }
            
            logger.info(f"Generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
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
    
    parser = argparse.ArgumentParser(description="RAG Q&A CLI")
    parser.add_argument("--question", "-q", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    try:
        qa = RAGQA()
        
        # Check collection
        info = qa.get_collection_info()
        if "error" in info:
            print(f"Error: {info['error']}")
            return 1
        
        print(f"RAG Q&A System Ready")
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

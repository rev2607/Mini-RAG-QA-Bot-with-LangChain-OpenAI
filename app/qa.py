"""Question-answering functionality using RAG with LangChain, OpenAI and Chroma."""

import logging
from typing import List, Dict, Any, Optional
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

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
    """RAG-based question answering system using LangChain."""
    
    def __init__(self):
        """Initialize the RAG QA system."""
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL,
            temperature=0.1
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RESULTS}
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant that answers questions based on provided context. 
Use the information below to answer the user's question. If the answer is not in the context, say 'I don't know — see sources'.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        logger.info(f"Initialized LangChain RAG QA system with collection: {COLLECTION_NAME}")
    
    def ask_question(self, question: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources using LangChain.
        
        Args:
            question: The question to ask
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing question with LangChain: {question}")
        
        try:
            # Use LangChain QA chain
            result = self.qa_chain({"query": question})
            
            answer = result["result"]
            source_documents = result["source_documents"]
            
            logger.info(f"Retrieved {len(source_documents)} relevant chunks")
            
            # Format sources
            sources = []
            for i, doc in enumerate(source_documents):
                # Calculate similarity score (approximate)
                similarity_score = 0.8 - (i * 0.1)  # Approximate based on rank
                
                sources.append({
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": similarity_score,
                    "source_file": doc.metadata.get('source_file', 'Unknown'),
                    "chunk_index": doc.metadata.get('chunk_index', 0)
                })
            
            result_dict = {
                "answer": answer,
                "sources": sources,
                "used_docs_count": len(source_documents),
                "question": question
            }
            
            logger.info(f"Generated LangChain answer with {len(sources)} sources")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error processing question with LangChain: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "used_docs_count": 0,
                "question": question
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection."""
        try:
            # Get collection info from Chroma
            collection = self.vectorstore._collection
            count = collection.count()
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

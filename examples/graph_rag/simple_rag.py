#!/usr/bin/env python3
"""
Simple Graph RAG Example

This example demonstrates how to use the knowledge graph built with our pipeline
to implement a Retrieval-Augmented Generation (RAG) system.
"""

import os
import sys
from typing import List, Dict, Any
import logging
import ray
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pipeline.graph.search import KnowledgeGraphSearch
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.path.expanduser(os.getenv("MODEL_DIR", "~/ray-models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

class GraphRAG:
    """
    A simple RAG implementation using Neo4j for retrieval and a local LLM for generation.
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init()
            
        logger.info("Initializing Graph RAG system...")
        
        # Initialize search component
        self.search = KnowledgeGraphSearch.remote()
        
        # Initialize embedding model for queries
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            logger.info(f"Embedding model loaded on {device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
        
        # Initialize LLM for generation
        logger.info("Loading LLM...")
        model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
        try:
            callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=MODEL_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n_ctx=CONTEXT_SIZE,
                callback_manager=callbacks,
                n_gpu_layers=-1,
                verbose=True
            )
            logger.info("LLM loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            raise
        
        # Template for RAG prompt
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided knowledge.

Context information from the knowledge graph:
{context}

User question: {question}

Instructions:
1. Answer the question using the information from the context
2. If the context doesn't contain enough information, say so and provide a general answer
3. Cite your sources by mentioning the article titles and sources

Your answer:
""")
        
        logger.info("Graph RAG system initialized!")
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks from the knowledge graph.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        embedding = self.get_embeddings(query)
        
        # Perform hybrid search
        results = ray.get(self.search.hybrid_search.remote(
            query=query,
            embedding=embedding,
            limit=limit,
            vector_weight=0.7
        ))
        
        return results
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the knowledge graph."
            
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.get('source', 'Unknown source')}"
            article_title = chunk.get('article_title', 'Untitled')
            article_url = chunk.get('article_url', '')
            
            # Format the chunk with metadata
            chunk_text = f"[{i+1}] From article '{article_title}' ({source_info}):\n{chunk['content']}\n"
            context_parts.append(chunk_text)
            
        return "\n".join(context_parts)
    
    def answer(self, question: str) -> str:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant chunks
        chunks = self.retrieve(question)
        
        # Format context
        context = self.format_context(chunks)
        
        # Generate answer
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        answer = self.llm(prompt)
        return answer

def main():
    """Run an interactive demo of the Graph RAG system."""
    print("\n🤖 Graph RAG Demo 🤖")
    print("===================")
    print("This demo uses the knowledge graph to answer your questions.")
    print("Type 'exit' to quit.\n")
    
    try:
        rag = GraphRAG()
        
        while True:
            question = input("\n🔍 Enter your question: ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            print("\n⏳ Searching knowledge graph and generating answer...\n")
            answer = rag.answer(question)
            
            print("\n✅ Answer:")
            print(answer)
            print("\n" + "-" * 80)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in RAG demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
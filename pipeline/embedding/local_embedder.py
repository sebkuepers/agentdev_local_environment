#!/usr/bin/env python3
"""
Local Text Embedder using Sentence Transformers

This module provides a class for generating embeddings using a local model
via the sentence-transformers library.
"""

import os
import numpy as np
from typing import List, Union, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import ray

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote
class LocalEmbedder:
    """
    A class for generating text embeddings using a local model.
    Designed to be used as a Ray actor for distributed processing.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name: Name of the Sentence Transformers model to use
            device: Device to use for computation (cpu, cuda, mps)
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        logger.info(f"Initializing embedder with model: {self.model_name}")
        
        # Automatically determine device if not specified
        if device is None:
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load the model
        try:
            import torch
            self.model = SentenceTransformer(self.model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Process in batches to avoid OOM issues
            all_embeddings = []
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + EMBEDDING_BATCH_SIZE]
                embeddings = self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
                # Convert to list of lists for serialization
                embeddings_list = embeddings.tolist()
                all_embeddings.extend(embeddings_list)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of document dictionaries.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of documents with added 'embedding' field
        """
        if not documents:
            return []
        
        try:
            # Extract text content
            texts = [doc.get("content", "") for doc in documents]
            # Generate embeddings
            embeddings = self.embed_texts(texts)
            
            # Add embeddings back to documents
            for doc, embedding in zip(documents, embeddings):
                doc["embedding"] = embedding
                
            return documents
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension size (int)
        """
        return self.dimension

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote embedder actor
    embedder = LocalEmbedder.remote()
    
    # Test with some example texts
    test_texts = [
        "This is an example sentence for embedding.",
        "Another example with different content.",
        "Ray makes distributed computing simple and efficient."
    ]
    
    # Get embeddings
    embeddings = ray.get(embedder.embed_texts.remote(test_texts))
    
    # Print results
    for i, embedding in enumerate(embeddings):
        print(f"Text {i+1} embedding shape: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Norm: {np.linalg.norm(embedding):.4f}")
        print("---")
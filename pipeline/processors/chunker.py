#!/usr/bin/env python3
"""
Text Chunker

This module provides functionality for splitting long text into smaller chunks
with overlap, suitable for embedding and storage in a knowledge graph.
"""

import os
import re
import uuid
from typing import List, Dict, Any
import logging
import ray
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

@ray.remote
class TextChunker:
    """
    A class for splitting documents into smaller chunks with overlap.
    Designed to be used as a Ray actor for distributed processing.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        logger.info(f"Initialized chunker with size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split a text into smaller chunks with overlap.
        
        Args:
            text: The text to split
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []
            
        # Prepare chunks
        chunks = []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split text into paragraphs
        paragraphs = [p for p in text.split("\n") if p.strip()]
        
        # Create chunks from paragraphs
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed the chunk size
            if current_size + len(para) > self.chunk_size and current_chunk:
                # Save the current chunk
                chunk_text = "\n".join(current_chunk)
                chunk_dict = self._create_chunk_dict(chunk_text, metadata)
                chunks.append(chunk_dict)
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunks = []
                
                # Add paragraphs from the end of the previous chunk until we reach desired overlap
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.chunk_overlap:
                        overlap_chunks.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_size = overlap_size
            
            # Add the current paragraph to the chunk
            current_chunk.append(para)
            current_size += len(para)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_dict = self._create_chunk_dict(chunk_text, metadata)
            chunks.append(chunk_dict)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _create_chunk_dict(self, chunk_text: str, metadata: Dict = None) -> Dict:
        """
        Create a dictionary for a chunk with metadata.
        
        Args:
            chunk_text: The text content of the chunk
            metadata: Additional metadata to include
            
        Returns:
            Chunk dictionary
        """
        chunk_id = str(uuid.uuid4())
        
        chunk_dict = {
            "id": chunk_id,
            "content": chunk_text,
        }
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key != "content" and key != "id":
                    chunk_dict[key] = value
        
        return chunk_dict
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process a batch of documents and split them into chunks.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        
        for doc in documents:
            try:
                # Extract content and metadata
                content = doc.get("content", "")
                
                # Create metadata from the document
                metadata = {k: v for k, v in doc.items() if k != "content" and k != "embedding"}
                
                # Add source document ID
                metadata["source_url"] = doc.get("url", "")
                metadata["source_title"] = doc.get("title", "")
                
                # Create chunks
                chunks = self.chunk_text(content, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document: {e}")
        
        return all_chunks

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote chunker actor
    chunker = TextChunker.remote()
    
    # Test with example document
    test_doc = {
        "url": "https://example.com/article",
        "title": "Example Article",
        "content": "This is a test article.\n\nIt has multiple paragraphs.\n\n" + "This is a long paragraph that should be included in the chunk. " * 20,
        "source": "Example Source",
        "publication_date": "2023-05-18"
    }
    
    # Get chunks
    chunks = ray.get(chunker.chunk_documents.remote([test_doc]))
    
    # Print results
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}")
        print(f"ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(f"Content length: {len(chunk['content'])} chars")
        print(f"First 100 chars: {chunk['content'][:100]}...")
        print("---")
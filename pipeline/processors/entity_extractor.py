#!/usr/bin/env python3
"""
Entity Extraction Module

This module uses neo4j-graphrag to extract entities and relationships from text.
It's designed to be used as a Ray actor for distributed processing.
"""

import os
import sys
import uuid
from typing import List, Dict, Any, Optional
import logging
import ray
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import GraphRAG components
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.extractors import BasicExtractor
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Entity and relationship types
from pipeline.graph.schema import ENTITY_TYPES, RELATION_TYPES

@ray.remote
class EntityExtractor:
    """
    A class for extracting entities and relationships from text.
    Designed to be used as a Ray actor for distributed processing.
    """
    
    def __init__(self, use_llm: bool = False, llm_api_key: Optional[str] = None):
        """
        Initialize the entity extractor.
        
        Args:
            use_llm: Whether to use an LLM for extraction (more accurate but slower)
            llm_api_key: API key for LLM service (if using an external LLM)
        """
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        # Initialize embeddings model
        try:
            self.embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embeddings model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
        
        # Initialize LLM-based extraction if requested
        self.use_llm = use_llm
        if self.use_llm and llm_api_key:
            try:
                from neo4j_graphrag.llm import OpenAILLM
                self.llm = OpenAILLM(api_key=llm_api_key)
                
                # Create KG builder with LLM
                self.kg_builder = SimpleKGPipeline(
                    llm=self.llm,
                    driver=self.driver,
                    embedder=self.embedder,
                    entities=ENTITY_TYPES,
                    relations=RELATION_TYPES
                )
                logger.info("Initialized LLM-based entity extraction pipeline")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}, falling back to rule-based extraction")
                self.use_llm = False
        
        if not self.use_llm:
            # Use rule-based extraction
            self.extractor = BasicExtractor(self.driver, entity_types=ENTITY_TYPES)
            logger.info("Initialized rule-based entity extraction")
    
    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def extract_entities(self, text: str, metadata: Dict = None) -> Dict[str, List[Dict]]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text to extract entities from
            metadata: Additional metadata
            
        Returns:
            Dictionary with entities and relationships
        """
        if not text:
            return {"entities": [], "relationships": []}
        
        try:
            # Add unique ID if not provided
            if not metadata:
                metadata = {}
            if "id" not in metadata:
                metadata["id"] = str(uuid.uuid4())
                
            # Extract entities and relationships
            if self.use_llm:
                # Use LLM-based extraction
                extraction_result = self.kg_builder.extract(
                    text=text,
                    metadata=metadata
                )
                return extraction_result
            else:
                # Use rule-based extraction
                entities = self.extractor.extract(text)
                
                # Generate embeddings for entities
                for entity in entities:
                    if "name" in entity:
                        entity["embedding"] = self.embedder.embed_query(entity["name"])
                
                # Simple relationship heuristic - entities in the same text are related
                relationships = []
                for i, entity1 in enumerate(entities[:-1]):
                    for entity2 in enumerate(entities[i+1:]):
                        rel_type = "RELATED_TO"  # Default relationship
                        relationships.append({
                            "source": entity1["name"],
                            "target": entity2["name"],
                            "type": rel_type,
                            "metadata": {"source_text": metadata.get("id", "")}
                        })
                
                return {
                    "entities": entities,
                    "relationships": relationships
                }
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "relationships": []}
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process a batch of documents and extract entities and relationships.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of documents with added 'entities' and 'relationships' fields
        """
        processed_docs = []
        
        for doc in documents:
            try:
                # Extract content and metadata
                content = doc.get("content", "")
                
                # Create metadata from the document
                metadata = {k: v for k, v in doc.items() if k != "content" and k != "embedding"}
                
                # Add source document ID
                metadata["source_url"] = doc.get("url", "")
                metadata["source_title"] = doc.get("title", "")
                
                # Extract entities and relationships
                extraction_result = self.extract_entities(content, metadata)
                
                # Add extraction results to document
                doc["entities"] = extraction_result.get("entities", [])
                doc["relationships"] = extraction_result.get("relationships", [])
                
                processed_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                # Add the original document without entities
                processed_docs.append(doc)
        
        return processed_docs

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote entity extractor actor
    extractor = EntityExtractor.remote()
    
    # Test with example document
    test_doc = {
        "url": "https://example.com/article",
        "title": "Google DeepMind Announces New AI Model",
        "content": "Google DeepMind has announced a new AI model called Gemini. The model was developed in London by a team led by Demis Hassabis. Gemini is designed to process multiple types of data including text, images, and video. It is expected to compete with OpenAI's GPT-4 model.",
        "source": "Tech News",
        "publication_date": "2023-12-01"
    }
    
    # Process the document
    processed_doc = ray.get(extractor.process_documents.remote([test_doc]))[0]
    
    # Print results
    print(f"Extracted {len(processed_doc.get('entities', []))} entities:")
    for entity in processed_doc.get("entities", []):
        print(f"  - {entity.get('type', 'Unknown')}: {entity.get('name', '')}")
    
    print(f"Extracted {len(processed_doc.get('relationships', []))} relationships:")
    for rel in processed_doc.get("relationships", []):
        print(f"  - {rel.get('source', '')} {rel.get('type', '')} {rel.get('target', '')}")
#!/usr/bin/env python3
"""
Knowledge Graph Builder

This module provides functionality for building and updating a knowledge graph in Neo4j.
It handles creation of nodes and relationships from processed articles, chunks, and entities.
It integrates with neo4j-graphrag for enhanced knowledge graph construction.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import ray
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import GraphRAG components
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.document import Document
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Entity and relationship types
from pipeline.graph.schema import ENTITY_TYPES, RELATION_TYPES

@ray.remote
class KnowledgeGraphBuilder:
    """
    A class for building and updating a knowledge graph in Neo4j.
    Designed to be used as a Ray actor for distributed processing.
    Integrates with neo4j-graphrag for enhanced knowledge graph construction.
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize the knowledge graph builder.
        
        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
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
            
        # Initialize GraphRAG KG builder
        try:
            self.kg_builder = SimpleKGPipeline(
                driver=self.driver,
                embedder=self.embedder,
                entities=ENTITY_TYPES,
                relations=RELATION_TYPES
            )
            logger.info("Initialized GraphRAG KG builder")
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG KG builder: {e}")
            self.kg_builder = None
    
    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def add_article(self, article: Dict) -> Optional[str]:
        """
        Add an article to the knowledge graph.
        
        Args:
            article: Article dictionary with metadata
            
        Returns:
            Article ID (URL) if successful, None otherwise
        """
        try:
            with self.driver.session() as session:
                # Check if article already exists
                result = session.run(
                    "MATCH (a:Article {url: $url}) RETURN a.url",
                    url=article.get("url")
                )
                
                if result.single():
                    logger.info(f"Article already exists: {article.get('url')}")
                    return article.get("url")
                
                # Create article node
                session.run(
                    """
                    CREATE (a:Article {
                        url: $url,
                        title: $title,
                        summary: $summary,
                        content: $content,
                        source: $source,
                        publication_date: $pub_date,
                        topic: $topic,
                        keywords: $keywords,
                        created_at: $created_at
                    })
                    """,
                    url=article.get("url", ""),
                    title=article.get("title", ""),
                    summary=article.get("summary", ""),
                    content=article.get("content", ""),
                    source=article.get("source", ""),
                    pub_date=article.get("publication_date", datetime.now().strftime("%Y-%m-%d")),
                    topic=article.get("topic", ""),
                    keywords=article.get("keywords", []),
                    created_at=datetime.now().isoformat()
                )
                
                logger.info(f"Added article: {article.get('url')}")
                return article.get("url")
                
        except Exception as e:
            logger.error(f"Error adding article to graph: {e}")
            return None
    
    def add_chunks(self, chunks: List[Dict], article_url: str) -> int:
        """
        Add text chunks to the knowledge graph and link to their source article.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            article_url: URL of the source article
            
        Returns:
            Number of chunks added
        """
        if not chunks or not article_url:
            return 0
            
        added_count = 0
        try:
            with self.driver.session() as session:
                for chunk in chunks:
                    # Ensure the chunk has required fields
                    if "id" not in chunk or "content" not in chunk:
                        logger.warning("Chunk missing required fields, skipping")
                        continue
                        
                    # Check if embedding is present
                    if "embedding" not in chunk:
                        logger.warning("Chunk missing embedding, skipping")
                        continue
                    
                    # Add the chunk node
                    result = session.run(
                        """
                        MATCH (a:Article {url: $article_url})
                        MERGE (c:Chunk {id: $id})
                        ON CREATE SET 
                            c.content = $content,
                            c.embedding = $embedding,
                            c.created_at = $created_at
                        WITH a, c
                        MERGE (a)-[:HAS_CHUNK]->(c)
                        RETURN c.id
                        """,
                        article_url=article_url,
                        id=chunk.get("id"),
                        content=chunk.get("content"),
                        embedding=chunk.get("embedding"),
                        created_at=datetime.now().isoformat()
                    )
                    
                    # Check if chunk was added
                    if result.single():
                        added_count += 1
                        
            logger.info(f"Added {added_count} chunks for article: {article_url}")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding chunks to graph: {e}")
            return added_count
            
    def add_entities(self, entities: List[Dict], relationships: List[Dict], article_url: str) -> Tuple[int, int]:
        """
        Add entities and relationships to the knowledge graph and link to their source article.
        
        Args:
            entities: List of entity dictionaries with embeddings
            relationships: List of relationship dictionaries
            article_url: URL of the source article
            
        Returns:
            Tuple of (entity_count, relationship_count) added
        """
        if not entities or not article_url:
            return (0, 0)
            
        entity_count = 0
        relationship_count = 0
        
        try:
            with self.driver.session() as session:
                # Add entities
                for entity in entities:
                    # Ensure the entity has required fields
                    if "name" not in entity or "type" not in entity:
                        logger.warning("Entity missing required fields, skipping")
                        continue
                        
                    # Add the entity node
                    entity_type = entity.get("type")
                    if entity_type not in ENTITY_TYPES:
                        entity_type = "Topic"  # Default to Topic if unknown type
                        
                    result = session.run(
                        f"""
                        MATCH (a:Article {{url: $article_url}})
                        MERGE (e:{entity_type} {{name: $name}})
                        ON CREATE SET 
                            e.description = $description,
                            e.embedding = $embedding,
                            e.created_at = $created_at
                        WITH a, e
                        MERGE (e)-[:MENTIONED_IN]->(a)
                        RETURN e.name
                        """,
                        article_url=article_url,
                        name=entity.get("name"),
                        description=entity.get("description", ""),
                        embedding=entity.get("embedding", []),
                        created_at=datetime.now().isoformat()
                    )
                    
                    # Check if entity was added
                    if result.single():
                        entity_count += 1
                
                # Add relationships
                for rel in relationships:
                    # Ensure the relationship has required fields
                    if "source" not in rel or "target" not in rel or "type" not in rel:
                        logger.warning("Relationship missing required fields, skipping")
                        continue
                        
                    # Add the relationship
                    rel_type = rel.get("type")
                    if rel_type not in RELATION_TYPES:
                        rel_type = "RELATED_TO"  # Default to RELATED_TO if unknown type
                        
                    result = session.run(
                        f"""
                        MATCH (e1) WHERE e1.name = $source
                        MATCH (e2) WHERE e2.name = $target
                        WHERE NOT e1 = e2
                        MERGE (e1)-[r:{rel_type}]->(e2)
                        ON CREATE SET 
                            r.metadata = $metadata,
                            r.created_at = $created_at
                        RETURN type(r)
                        """,
                        source=rel.get("source"),
                        target=rel.get("target"),
                        metadata=rel.get("metadata", {}),
                        created_at=datetime.now().isoformat()
                    )
                    
                    # Check if relationship was added
                    if result.single():
                        relationship_count += 1
                        
            logger.info(f"Added {entity_count} entities and {relationship_count} relationships for article: {article_url}")
            return (entity_count, relationship_count)
            
        except Exception as e:
            logger.error(f"Error adding entities and relationships to graph: {e}")
            return (entity_count, relationship_count)
    
    def add_article_with_chunks(self, article: Dict, chunks: List[Dict]) -> bool:
        """
        Add an article and its chunks to the knowledge graph in a single transaction.
        
        Args:
            article: Article dictionary
            chunks: List of chunk dictionaries with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add the article
            article_url = self.add_article(article)
            
            if not article_url:
                return False
                
            # Add chunks and link to article
            if chunks:
                self.add_chunks(chunks, article_url)
                
            # Add entities and relationships if they exist
            if "entities" in article and "relationships" in article:
                self.add_entities(
                    article.get("entities", []),
                    article.get("relationships", []),
                    article_url
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding article with chunks: {e}")
            return False
    
    def add_document_with_graphrag(self, document: Dict) -> bool:
        """
        Use neo4j-graphrag to add a document to the knowledge graph.
        This method extracts entities, relationships and creates the graph structure.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.kg_builder:
            logger.warning("GraphRAG KG builder not available, falling back to standard method")
            return self.add_article_with_chunks(document, [])
            
        try:
            # Create GraphRAG Document object
            doc = Document(
                id=document.get("url", ""),
                text=document.get("content", ""),
                metadata={
                    "title": document.get("title", ""),
                    "source": document.get("source", ""),
                    "publication_date": document.get("publication_date", ""),
                    "url": document.get("url", "")
                }
            )
            
            # Add document to knowledge graph
            result = self.kg_builder.add_document(doc)
            
            if result:
                logger.info(f"Added document using GraphRAG: {document.get('url')}")
                return True
            else:
                logger.warning(f"Failed to add document using GraphRAG: {document.get('url')}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding document with GraphRAG: {e}")
            return False
    
    def create_topic_relationships(self) -> int:
        """
        Create RELATED_TO relationships between articles with the same topic.
        
        Returns:
            Number of relationships created
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a1:Article), (a2:Article)
                    WHERE a1.topic = a2.topic AND a1.url <> a2.url
                    AND NOT (a1)-[:RELATED_TO]-(a2)
                    WITH DISTINCT a1, a2
                    CREATE (a1)-[r:RELATED_TO {reason: 'same_topic'}]->(a2)
                    RETURN count(r) as count
                    """
                )
                
                count = result.single()["count"]
                logger.info(f"Created {count} RELATED_TO relationships based on topics")
                return count
                
        except Exception as e:
            logger.error(f"Error creating topic relationships: {e}")
            return 0
    
    def get_article_count(self) -> int:
        """
        Get the total number of articles in the knowledge graph.
        
        Returns:
            Number of articles
        """
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (a:Article) RETURN count(a) as count")
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting article count: {e}")
            return 0
    
    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the knowledge graph.
        
        Returns:
            Number of chunks
        """
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
            
    def get_entity_count(self) -> int:
        """
        Get the total number of entities in the knowledge graph.
        
        Returns:
            Number of entities
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e)
                    WHERE any(label IN labels(e) WHERE label IN $entity_types)
                    RETURN count(e) as count
                    """,
                    entity_types=ENTITY_TYPES
                )
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting entity count: {e}")
            return 0
            
    def get_relationship_count(self) -> int:
        """
        Get the total number of relationships in the knowledge graph.
        
        Returns:
            Number of relationships
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE type(r) IN $relation_types
                    RETURN count(r) as count
                    """,
                    relation_types=RELATION_TYPES
                )
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting relationship count: {e}")
            return 0

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote graph builder actor
    builder = KnowledgeGraphBuilder.remote()
    
    # Get current counts
    article_count = ray.get(builder.get_article_count.remote())
    chunk_count = ray.get(builder.get_chunk_count.remote())
    entity_count = ray.get(builder.get_entity_count.remote())
    relationship_count = ray.get(builder.get_relationship_count.remote())
    
    print(f"Current knowledge graph stats:")
    print(f"  Articles: {article_count}")
    print(f"  Chunks: {chunk_count}")
    print(f"  Entities: {entity_count}")
    print(f"  Relationships: {relationship_count}")
    
    # Example of adding a test article
    test_article = {
        "url": "https://example.com/test-article",
        "title": "Google DeepMind Announces New AI Model",
        "content": "Google DeepMind has announced a new AI model called Gemini. The model was developed in London by a team led by Demis Hassabis. Gemini is designed to process multiple types of data including text, images, and video. It is expected to compete with OpenAI's GPT-4 model.",
        "summary": "Google DeepMind announces Gemini AI model.",
        "source": "Tech News",
        "publication_date": "2023-12-01",
        "topic": "AI",
        "keywords": ["AI", "Gemini", "Google", "DeepMind"]
    }
    
    # Add using GraphRAG
    print("Adding document using GraphRAG...")
    success = ray.get(builder.add_document_with_graphrag.remote(test_article))
    
    if success:
        print("Successfully added document with GraphRAG!")
    else:
        print("Failed to add document with GraphRAG, falling back to standard method...")
#!/usr/bin/env python3
"""
Knowledge Graph Builder

This module provides functionality for building and updating a knowledge graph in Neo4j.
It handles creation of nodes and relationships from processed articles and chunks.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import ray
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

@ray.remote
class KnowledgeGraphBuilder:
    """
    A class for building and updating a knowledge graph in Neo4j.
    Designed to be used as a Ray actor for distributed processing.
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
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding article with chunks: {e}")
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
    
    print(f"Current knowledge graph stats:")
    print(f"  Articles: {article_count}")
    print(f"  Chunks: {chunk_count}")
    
    # Example of adding a test article
    test_article = {
        "url": "https://example.com/test-article",
        "title": "Test Article",
        "content": "This is a test article content.",
        "summary": "Test summary",
        "source": "Test Source",
        "publication_date": "2023-05-18",
        "topic": "testing",
        "keywords": ["test", "example"]
    }
    
    # Example chunk with fake embedding
    test_chunk = {
        "id": "test-chunk-1",
        "content": "This is a test chunk content.",
        "embedding": [0.1] * 384  # Fake embedding vector
    }
    
    # Add the article and chunk
    success = ray.get(builder.add_article_with_chunks.remote(test_article, [test_chunk]))
    
    if success:
        print("Successfully added test article and chunk!")
    else:
        print("Failed to add test article and chunk.")
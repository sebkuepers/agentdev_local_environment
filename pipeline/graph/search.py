#!/usr/bin/env python3
"""
Knowledge Graph Search

This module provides functionality for semantic search in the Neo4j knowledge graph.
It supports vector search, full-text search, and hybrid search capabilities.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
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
class KnowledgeGraphSearch:
    """
    A class for searching the knowledge graph in Neo4j.
    Designed to be used as a Ray actor for distributed search.
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize the knowledge graph search.
        
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
    
    def vector_search(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Perform a vector search to find similar chunks.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embedding, $embedding) AS score
                    WHERE score > 0.7
                    RETURN c.id AS id, c.content AS content, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    embedding=embedding,
                    limit=limit
                )
                
                chunks = []
                for record in result:
                    chunks.append({
                        "id": record["id"],
                        "content": record["content"],
                        "score": record["score"]
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []
    
    def text_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform a full-text search on chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with score
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes("chunk_search", $query) 
                    YIELD node, score
                    RETURN node.id AS id, node.content AS content, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit
                )
                
                chunks = []
                for record in result:
                    chunks.append({
                        "id": record["id"],
                        "content": record["content"],
                        "score": record["score"]
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error performing text search: {e}")
            return []
    
    def hybrid_search(self, query: str, embedding: List[float], 
                     limit: int = 5, vector_weight: float = 0.7) -> List[Dict]:
        """
        Perform a hybrid search combining vector and full-text search.
        
        Args:
            query: Search query string
            embedding: Query embedding vector
            limit: Maximum number of results to return
            vector_weight: Weight given to vector search (0-1)
            
        Returns:
            List of chunk dictionaries with combined scores
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    // Vector search
                    MATCH (c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embedding, $embedding) AS vectorScore
                    
                    // Text search score calculation
                    WITH c, vectorScore
                    CALL {
                        WITH c
                        CALL db.index.fulltext.queryNodes("chunk_search", $query) 
                        YIELD node, score
                        WHERE node.id = c.id
                        RETURN score as textScore
                    }
                    
                    // Calculate combined score
                    WITH c, 
                         vectorScore * $vectorWeight + 
                         CASE WHEN textScore IS NULL THEN 0 ELSE textScore * (1 - $vectorWeight) END 
                         AS combinedScore
                    WHERE combinedScore > 0
                    
                    // Get related article
                    OPTIONAL MATCH (a:Article)-[:HAS_CHUNK]->(c)
                    
                    RETURN 
                        c.id AS id, 
                        c.content AS content, 
                        combinedScore AS score,
                        a.url AS article_url,
                        a.title AS article_title,
                        a.source AS source
                    ORDER BY combinedScore DESC
                    LIMIT $limit
                    """,
                    query=query,
                    embedding=embedding,
                    limit=limit,
                    vectorWeight=vector_weight
                )
                
                chunks = []
                for record in result:
                    chunks.append({
                        "id": record["id"],
                        "content": record["content"],
                        "score": record["score"],
                        "article_url": record["article_url"],
                        "article_title": record["article_title"],
                        "source": record["source"]
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    def get_article_context(self, chunk_id: str) -> Dict:
        """
        Get the article context for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Article metadata dictionary
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
                    RETURN a.url AS url, a.title AS title, a.source AS source, 
                           a.publication_date AS date
                    """,
                    chunk_id=chunk_id
                )
                
                record = result.single()
                if record:
                    return {
                        "url": record["url"],
                        "title": record["title"],
                        "source": record["source"],
                        "date": record["date"]
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting article context: {e}")
            return {}
    
    def get_related_articles(self, article_url: str, limit: int = 3) -> List[Dict]:
        """
        Get related articles for a given article.
        
        Args:
            article_url: URL of the article
            limit: Maximum number of results to return
            
        Returns:
            List of related article dictionaries
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a1:Article {url: $url})-[:RELATED_TO]-(a2:Article)
                    RETURN a2.url AS url, a2.title AS title, a2.source AS source, 
                           a2.publication_date AS date, a2.topic AS topic
                    LIMIT $limit
                    """,
                    url=article_url,
                    limit=limit
                )
                
                articles = []
                for record in result:
                    articles.append({
                        "url": record["url"],
                        "title": record["title"],
                        "source": record["source"],
                        "date": record["date"],
                        "topic": record["topic"]
                    })
                
                return articles
                
        except Exception as e:
            logger.error(f"Error getting related articles: {e}")
            return []

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote search actor
    search = KnowledgeGraphSearch.remote()
    
    # Example text search
    query = "language models"
    results = ray.get(search.text_search.remote(query))
    
    print(f"Text search results for '{query}':")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        print("---")
#!/usr/bin/env python3
"""
Knowledge Graph Search

This module provides functionality for semantic search in the Neo4j knowledge graph.
It supports vector search, full-text search, hybrid search, and graph-enhanced 
retrieval capabilities through neo4j-graphrag.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import ray
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import GraphRAG components
from neo4j_graphrag.experimental.retriever import VectorRetriever
from neo4j_graphrag.experimental.retriever import GraphRetriever
from neo4j_graphrag.experimental.retrieval import GraphRAG
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.document import Document

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
class KnowledgeGraphSearch:
    """
    A class for searching the knowledge graph in Neo4j.
    Designed to be used as a Ray actor for distributed search.
    Integrates with neo4j-graphrag for enhanced retrieval.
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
            
        # Initialize embeddings model
        try:
            self.embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embeddings model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
            
        # Initialize GraphRAG components
        try:
            # Vector retrieval for chunks
            self.chunk_retriever = VectorRetriever(
                driver=self.driver,
                index_name="chunk_embedding",
                embedder=self.embedder,
                node_labels=["Chunk"],
                embedding_field="embedding",
                content_field="content"
            )
            
            # Vector retrieval for entities
            self.entity_retriever = VectorRetriever(
                driver=self.driver,
                index_name="entity_embedding",
                embedder=self.embedder,
                node_labels=ENTITY_TYPES,
                embedding_field="embedding",
                content_field="name"
            )
            
            # Graph retrieval (entity-based)
            self.graph_retriever = GraphRetriever(
                driver=self.driver,
                entity_types=ENTITY_TYPES,
                relation_types=RELATION_TYPES
            )
            
            # Combined GraphRAG retriever
            self.graphrag = GraphRAG(
                retriever=self.chunk_retriever,
                embedder=self.embedder
            )
            
            logger.info("Initialized GraphRAG search components")
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG search components: {e}")
            self.chunk_retriever = None
            self.entity_retriever = None
            self.graph_retriever = None
            self.graphrag = None
    
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
        # Try to use GraphRAG retriever first
        if self.chunk_retriever:
            try:
                results = self.chunk_retriever.retrieve(
                    query_embedding=embedding,
                    top_k=limit
                )
                
                # Format results to match expected output
                chunks = []
                for item in results:
                    chunks.append({
                        "id": item.metadata.get("id", ""),
                        "content": item.content,
                        "score": item.score,
                        "article_url": item.metadata.get("article_url", ""),
                        "article_title": item.metadata.get("title", "")
                    })
                
                if chunks:
                    return chunks
            except Exception as e:
                logger.warning(f"GraphRAG vector search failed, falling back to standard: {e}")
        
        # Fall back to standard vector search
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
        # Try to use GraphRAG retriever first
        if self.graphrag and self.chunk_retriever:
            try:
                results = self.graphrag.search(
                    query_text=query,
                    retriever_config={"top_k": limit}
                )
                
                # Process results
                if results and results.passages:
                    chunks = []
                    for passage in results.passages:
                        chunks.append({
                            "id": passage.metadata.get("id", ""),
                            "content": passage.content,
                            "score": passage.score,
                            "article_url": passage.metadata.get("article_url", ""),
                            "article_title": passage.metadata.get("article_title", ""),
                            "source": passage.metadata.get("source", "")
                        })
                    
                    return chunks
            except Exception as e:
                logger.warning(f"GraphRAG hybrid search failed, falling back to standard: {e}")
        
        # Fall back to standard hybrid search
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
    
    def graph_enhanced_search(self, query: str, embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Perform a graph-enhanced search that uses entity relationships.
        
        Args:
            query: Search query string
            embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with scores enhanced by graph context
        """
        if not self.graph_retriever or not self.entity_retriever:
            logger.warning("Graph retriever not available, falling back to hybrid search")
            return self.hybrid_search(query, embedding, limit)
            
        try:
            # First, get relevant entities
            entity_results = self.entity_retriever.retrieve(
                query_embedding=embedding,
                top_k=max(3, limit // 2)
            )
            
            if not entity_results:
                logger.warning("No entities found, falling back to hybrid search")
                return self.hybrid_search(query, embedding, limit)
                
            # Extract entity IDs
            entity_ids = [entity.id for entity in entity_results if entity.id]
            
            # Use graph retrieval to find related chunks
            graph_results = self.graph_retriever.retrieve(
                entity_ids=entity_ids,
                query_embedding=embedding,
                top_k=limit
            )
            
            # Process and return results
            chunks = []
            for item in graph_results:
                chunks.append({
                    "id": item.metadata.get("id", ""),
                    "content": item.content,
                    "score": item.score,
                    "article_url": item.metadata.get("article_url", ""),
                    "article_title": item.metadata.get("article_title", ""),
                    "source": item.metadata.get("source", ""),
                    "entity_context": item.metadata.get("entity_context", "")
                })
            
            # If graph retrieval returned nothing, fall back to hybrid search
            if not chunks:
                logger.warning("Graph retrieval returned no results, falling back to hybrid search")
                return self.hybrid_search(query, embedding, limit)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error performing graph-enhanced search: {e}")
            return self.hybrid_search(query, embedding, limit)
    
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
            
    def get_entity_information(self, entity_name: str) -> Dict:
        """
        Get information about an entity, including relationships.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Entity information dictionary
        """
        try:
            with self.driver.session() as session:
                # Get entity information
                entity_result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name = $name AND any(label IN labels(e) WHERE label IN $entity_types)
                    RETURN
                        e.name AS name,
                        labels(e) AS types,
                        e.description AS description
                    LIMIT 1
                    """,
                    name=entity_name,
                    entity_types=ENTITY_TYPES
                )
                
                entity_record = entity_result.single()
                if not entity_record:
                    return {}
                    
                # Get entity relationships
                rel_result = session.run(
                    """
                    MATCH (e)-[r]-(other)
                    WHERE e.name = $name 
                    AND any(label IN labels(e) WHERE label IN $entity_types)
                    AND any(label IN labels(other) WHERE label IN $entity_types)
                    RETURN 
                        type(r) AS relation_type,
                        other.name AS related_entity,
                        labels(other) AS related_entity_types,
                        CASE WHEN startNode(r) = e THEN "outgoing" ELSE "incoming" END AS direction
                    LIMIT 10
                    """,
                    name=entity_name,
                    entity_types=ENTITY_TYPES
                )
                
                relationships = []
                for rel in rel_result:
                    relationships.append({
                        "relation_type": rel["relation_type"],
                        "entity": rel["related_entity"],
                        "entity_type": rel["related_entity_types"][0] if rel["related_entity_types"] else "Unknown",
                        "direction": rel["direction"]
                    })
                
                # Get mentioned in articles
                mentioned_result = session.run(
                    """
                    MATCH (e)-[:MENTIONED_IN]->(a:Article)
                    WHERE e.name = $name 
                    AND any(label IN labels(e) WHERE label IN $entity_types)
                    RETURN 
                        a.url AS url,
                        a.title AS title,
                        a.source AS source,
                        a.publication_date AS date
                    LIMIT 5
                    """,
                    name=entity_name,
                    entity_types=ENTITY_TYPES
                )
                
                articles = []
                for article in mentioned_result:
                    articles.append({
                        "url": article["url"],
                        "title": article["title"],
                        "source": article["source"],
                        "date": article["date"]
                    })
                
                # Return combined entity information
                return {
                    "name": entity_record["name"],
                    "type": entity_record["types"][0] if entity_record["types"] else "Unknown",
                    "description": entity_record["description"],
                    "relationships": relationships,
                    "articles": articles
                }
                
        except Exception as e:
            logger.error(f"Error getting entity information: {e}")
            return {}

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote search actor
    search = KnowledgeGraphSearch.remote()
    
    # Example of GraphRAG search
    query = "language models"
    
    # Get embedding for query
    from pipeline.embedding.local_embedder import LocalEmbedder
    embedder = LocalEmbedder.remote()
    embedding = ray.get(embedder.embed_texts.remote([query]))[0]
    
    # Perform graph-enhanced search
    results = ray.get(search.graph_enhanced_search.remote(query, embedding, 5))
    
    print(f"Graph-enhanced search results for '{query}':")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        if "entity_context" in result and result["entity_context"]:
            print(f"   Entity context: {result['entity_context']}")
        print("---")
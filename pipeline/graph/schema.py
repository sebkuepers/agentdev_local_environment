#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Schema Initialization

This script initializes the Neo4j database with the schema needed for the knowledge graph.
It creates indexes, constraints, and vector search capabilities.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

def initialize_schema():
    """Initialize the Neo4j database schema for the knowledge graph."""
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            # Create constraints and indexes for Article nodes
            logger.info("Creating constraints for Article nodes...")
            session.run("""
                CREATE CONSTRAINT article_id IF NOT EXISTS
                FOR (a:Article)
                REQUIRE a.url IS UNIQUE
            """)
            
            # Create constraints and indexes for Chunk nodes
            logger.info("Creating constraints for Chunk nodes...")
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk)
                REQUIRE c.id IS UNIQUE
            """)
            
            # Create indexes for topic and source
            logger.info("Creating indexes for efficient lookups...")
            session.run("CREATE INDEX article_topic IF NOT EXISTS FOR (a:Article) ON (a.topic)")
            session.run("CREATE INDEX article_source IF NOT EXISTS FOR (a:Article) ON (a.source)")
            session.run("CREATE INDEX article_date IF NOT EXISTS FOR (a:Article) ON (a.publication_date)")
            
            # Create vector index for embeddings
            logger.info(f"Creating vector index with dimension {EMBEDDING_DIMENSION}...")
            try:
                # First, check if the index already exists
                result = session.run("SHOW INDEXES WHERE name = 'chunk_embedding'")
                if not result.peek():
                    # Create the vector index
                    session.run(f"""
                        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                        FOR (c:Chunk)
                        ON (c.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIMENSION},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info("Vector index created successfully.")
                else:
                    logger.info("Vector index already exists.")
            except Exception as e:
                logger.error(f"Error creating vector index: {e}")
                
            # Create full-text search index
            logger.info("Creating full-text search index...")
            try:
                session.run("""
                    CREATE FULLTEXT INDEX article_search IF NOT EXISTS
                    FOR (a:Article)
                    ON EACH [a.title, a.content]
                """)
                session.run("""
                    CREATE FULLTEXT INDEX chunk_search IF NOT EXISTS
                    FOR (c:Chunk)
                    ON EACH [c.content]
                """)
                logger.info("Full-text search indexes created successfully.")
            except Exception as e:
                logger.error(f"Error creating full-text search index: {e}")
            
            logger.info("Schema initialization complete!")
            
            # Return summary of database
            result = session.run("""
                MATCH (n)
                RETURN DISTINCT labels(n) AS NodeTypes, count(n) AS Count
                UNION
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) AS NodeTypes, count(r) AS Count
            """)
            
            logger.info("Current database contents:")
            for record in result:
                logger.info(f"  {record['NodeTypes']}: {record['Count']}")

if __name__ == "__main__":
    try:
        logger.info("Initializing Neo4j knowledge graph schema...")
        initialize_schema()
        logger.info("Schema initialization completed successfully!")
    except Exception as e:
        logger.error(f"Error during schema initialization: {e}")
        sys.exit(1)
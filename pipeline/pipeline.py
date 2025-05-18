#!/usr/bin/env python3
"""
Ray-based Data Pipeline for Knowledge Graph Construction

This module implements the main data pipeline that:
1. Crawls articles from RSS feeds
2. Chunks the articles into smaller pieces
3. Generates embeddings for the chunks
4. Stores everything in the Neo4j knowledge graph
"""

import os
import sys
import time
from typing import List, Dict, Optional
import logging
from datetime import datetime
import ray
from dotenv import load_dotenv

# Import pipeline components
from pipeline.crawlers.rss_crawler import RSSCrawler
from pipeline.processors.chunker import TextChunker
from pipeline.embedding.local_embedder import LocalEmbedder
from pipeline.graph.builder import KnowledgeGraphBuilder
from pipeline.graph.schema import initialize_schema

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RSS_SOURCES = os.getenv("RSS_SOURCES", "").split(",")
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "50"))

class Pipeline:
    """
    A distributed data pipeline for building and updating a knowledge graph.
    """
    
    def __init__(self):
        """Initialize the pipeline and Ray actors."""
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init()
            
        logger.info("Initializing pipeline components...")
        
        # Create Ray actors for each pipeline component
        self.crawler = RSSCrawler.remote()
        self.chunker = TextChunker.remote()
        self.embedder = LocalEmbedder.remote()
        self.graph_builder = KnowledgeGraphBuilder.remote()
        
        logger.info("Pipeline initialized!")
    
    def run(self, sources: List[str] = None):
        """
        Run the complete pipeline.
        
        Args:
            sources: List of RSS feed URLs (default: from env)
        """
        start_time = time.time()
        logger.info("Starting pipeline run...")
        
        if sources is None:
            sources = RSS_SOURCES
            
        # Skip empty URLs
        sources = [url for url in sources if url and url.strip()]
        
        if not sources:
            logger.warning("No RSS sources provided. Check your RSS_SOURCES environment variable.")
            return
            
        # Step 1: Crawl articles
        logger.info(f"Fetching articles from {len(sources)} RSS sources...")
        article_refs = []
        for source in sources:
            article_ref = self.crawler.get_articles_from_feed.remote(
                source, max_articles=MAX_ARTICLES_PER_SOURCE
            )
            article_refs.append(article_ref)
        
        # Get all article results
        articles = []
        for article_batch in ray.get(article_refs):
            articles.extend(article_batch)
            
        logger.info(f"Fetched {len(articles)} articles.")
        
        if not articles:
            logger.warning("No articles found. Exiting pipeline.")
            return
            
        # Process each article
        processed_count = 0
        for article in articles:
            try:
                # Step 2: Chunk the article
                chunks_ref = self.chunker.chunk_text.remote(
                    article.get("content", ""),
                    {
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "publication_date": article.get("publication_date", "")
                    }
                )
                
                # Step 3: Generate embeddings for chunks
                chunks = ray.get(chunks_ref)
                if chunks:
                    chunks_with_embeddings_ref = self.embedder.embed_documents.remote(chunks)
                    chunks_with_embeddings = ray.get(chunks_with_embeddings_ref)
                    
                    # Step 4: Add to knowledge graph
                    success = ray.get(self.graph_builder.add_article_with_chunks.remote(
                        article, chunks_with_embeddings
                    ))
                    
                    if success:
                        processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing article {article.get('url', '')}: {e}")
                continue
                
        # Step 5: Create relationships between articles
        logger.info("Creating relationships between articles...")
        relationship_count = ray.get(self.graph_builder.create_topic_relationships.remote())
        
        # Log statistics
        article_count = ray.get(self.graph_builder.get_article_count.remote())
        chunk_count = ray.get(self.graph_builder.get_chunk_count.remote())
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds!")
        logger.info(f"Processed {processed_count}/{len(articles)} articles successfully.")
        logger.info(f"Created {relationship_count} relationships between articles.")
        logger.info(f"Knowledge graph now contains {article_count} articles and {chunk_count} chunks.")

def run_pipeline():
    """Run the full pipeline as a standalone function."""
    try:
        # Ensure the Neo4j schema is initialized
        initialize_schema()
        
        # Create and run the pipeline
        pipeline = Pipeline()
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
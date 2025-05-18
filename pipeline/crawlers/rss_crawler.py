#!/usr/bin/env python3
"""
RSS Feed Crawler

This module provides functionality to crawl RSS feeds and extract articles.
It uses feedparser to parse RSS feeds and newspaper3k to extract article content.
"""

import os
import time
import feedparser
import newspaper
from newspaper import Article
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
import logging
import ray
from dotenv import load_dotenv
import html2text

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RSS_SOURCES = os.getenv("RSS_SOURCES", "").split(",")
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "50"))

@ray.remote
class RSSCrawler:
    """
    A crawler for RSS feeds that extracts articles.
    Designed to be used as a Ray actor for distributed processing.
    """
    
    def __init__(self):
        """Initialize the RSS crawler."""
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0  # No wrapping
    
    def get_articles_from_feed(self, feed_url: str, max_articles: int = None) -> List[Dict]:
        """
        Extract articles from an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            max_articles: Maximum number of articles to extract (default: from env)
            
        Returns:
            List of article dictionaries
        """
        if max_articles is None:
            max_articles = MAX_ARTICLES_PER_SOURCE
            
        try:
            logger.info(f"Parsing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            # Extract the source name from the feed URL or title
            source_name = feed.feed.get('title', urlparse(feed_url).netloc)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                try:
                    article_url = entry.link
                    
                    # Extract publication date
                    if hasattr(entry, 'published_parsed'):
                        pub_date = time.strftime('%Y-%m-%d', entry.published_parsed)
                    else:
                        pub_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # Get article content
                    article_dict = self._extract_article_content(
                        url=article_url,
                        title=entry.get('title', ''),
                        summary=entry.get('summary', ''),
                        source=source_name,
                        pub_date=pub_date
                    )
                    
                    if article_dict:
                        articles.append(article_dict)
                        
                except Exception as e:
                    logger.warning(f"Error processing article {entry.get('link', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Extracted {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error processing feed {feed_url}: {e}")
            return []
    
    def _extract_article_content(self, url: str, title: str, summary: str, 
                               source: str, pub_date: str) -> Optional[Dict]:
        """
        Extract content from an article using newspaper3k.
        
        Args:
            url: URL of the article
            title: Title of the article
            summary: Summary from the RSS feed
            source: Source name of the article
            pub_date: Publication date
            
        Returns:
            Dictionary with article data or None if extraction failed
        """
        try:
            # Download and parse article
            article = Article(url)
            article.download()
            article.parse()
            
            # Use article title from newspaper if not available in RSS
            if not title and article.title:
                title = article.title
                
            # Get the text content
            content = article.text
            
            # If newspaper couldn't extract text properly, try using the summary
            if not content or len(content) < 100:
                # Try to convert HTML summary to text
                if summary and '<' in summary:
                    content = self.html_converter.handle(summary)
                else:
                    content = summary
            
            # Skip if we still don't have useful content
            if not content or len(content) < 50:
                logger.warning(f"Insufficient content for {url}, skipping")
                return None
                
            # Get keywords and topics
            article.nlp()
            keywords = article.keywords
            
            # Determine main topic from keywords
            topic = None
            if keywords and len(keywords) > 0:
                topic = keywords[0]
            
            # Create article dictionary
            article_dict = {
                "url": url,
                "title": title,
                "content": content,
                "summary": summary,
                "source": source,
                "publication_date": pub_date,
                "topic": topic,
                "keywords": keywords
            }
            
            return article_dict
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            
            # Create a minimal article from RSS data if we can't download it
            if title and (summary or len(summary) > 50):
                return {
                    "url": url,
                    "title": title,
                    "content": summary,
                    "summary": summary,
                    "source": source,
                    "publication_date": pub_date,
                    "topic": None,
                    "keywords": []
                }
            return None
            
    def get_articles_from_all_feeds(self, feed_urls: List[str] = None) -> List[Dict]:
        """
        Extract articles from multiple RSS feeds.
        
        Args:
            feed_urls: List of RSS feed URLs (default: from env)
            
        Returns:
            List of article dictionaries
        """
        if feed_urls is None:
            feed_urls = RSS_SOURCES
            
        # Skip empty URLs
        feed_urls = [url for url in feed_urls if url.strip()]
        
        if not feed_urls:
            logger.warning("No RSS feeds provided. Check your RSS_SOURCES environment variable.")
            return []
            
        all_articles = []
        for feed_url in feed_urls:
            try:
                articles = self.get_articles_from_feed(feed_url)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error processing feed {feed_url}: {e}")
                
        return all_articles

# Standalone usage example
if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create a remote crawler actor
    crawler = RSSCrawler.remote()
    
    # Test with example feed
    test_feed = "https://openai.com/blog/rss.xml"
    articles = ray.get(crawler.get_articles_from_feed.remote(test_feed, 3))
    
    # Print results
    for i, article in enumerate(articles):
        print(f"Article {i+1}: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Source: {article['source']}")
        print(f"Date: {article['publication_date']}")
        print(f"Content length: {len(article['content'])} chars")
        print(f"Topic: {article['topic']}")
        print("---")
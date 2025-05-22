#!/usr/bin/env python3
"""
Test script for GraphRAG integration.

This script tests the new GraphRAG capabilities by adding a test document to the 
knowledge graph and performing various types of searches.
"""

import os
import sys
import ray
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import our components
from pipeline.graph.builder import KnowledgeGraphBuilder
from pipeline.graph.search import KnowledgeGraphSearch
from pipeline.processors.entity_extractor import EntityExtractor
from pipeline.embedding.local_embedder import LocalEmbedder
from pipeline.processors.chunker import TextChunker

# Load environment variables
load_dotenv()

# Example test documents
TEST_DOCS = [
    {
        "url": "https://example.com/article1",
        "title": "OpenAI Releases GPT-5",
        "content": """
        OpenAI has announced the release of GPT-5, the latest version of its advanced language model. 
        The new model, developed by a team led by Sam Altman and Ilya Sutskever in San Francisco, 
        demonstrates significant improvements in reasoning, coding, and multimodal capabilities.
        
        According to OpenAI, GPT-5 outperforms previous models on benchmark tests by nearly 30%. 
        Microsoft, an investor in OpenAI, plans to integrate GPT-5 into its products including Bing
        and Microsoft Office applications.
        
        Critics, including Elon Musk, have expressed concerns about the rapid advancement of AI
        capabilities without sufficient safety measures in place. The European Union is considering
        new regulations in response to these developments.
        
        The model will be available through OpenAI's API starting next month, with pricing details
        to be announced soon.
        """,
        "source": "Tech News Daily",
        "publication_date": "2025-04-30",
        "topic": "Artificial Intelligence",
        "keywords": ["GPT-5", "OpenAI", "AI", "language model"]
    },
    {
        "url": "https://example.com/article2",
        "title": "Google DeepMind Unveils Gemini Pro 2",
        "content": """
        Google DeepMind has unveiled Gemini Pro 2, a significant upgrade to its multimodal AI system.
        The new model, developed at DeepMind's London headquarters under Demis Hassabis's leadership,
        shows impressive capabilities in handling complex reasoning tasks and multimodal inputs.
        
        Gemini Pro 2 is particularly notable for its ability to reason about images, videos, and text
        in an integrated way. The model has been trained on a diverse dataset and shows improved
        performance on benchmarks related to scientific reasoning and code generation.
        
        Google plans to integrate the model into its search engine and other products, potentially
        changing how users interact with Google services. The company stated that the model will
        be available to developers through the Google Cloud AI platform.
        
        The announcement comes as competition in the AI space intensifies, with OpenAI and other
        competitors also making significant advancements in the field.
        """,
        "source": "AI Insights",
        "publication_date": "2025-05-10",
        "topic": "Artificial Intelligence",
        "keywords": ["Gemini Pro", "Google DeepMind", "AI", "multimodal"]
    }
]

def initialize_ray():
    """Initialize Ray if not already running."""
    if not ray.is_initialized():
        ray.init()
        print("Ray initialized.")

def process_document(document):
    """Process a document through the pipeline components."""
    # Create Ray actors
    chunker = TextChunker.remote()
    embedder = LocalEmbedder.remote()
    extractor = EntityExtractor.remote()
    graph_builder = KnowledgeGraphBuilder.remote()
    
    print(f"Processing document: {document['title']}")
    
    # Step 1: Extract entities and relationships
    doc_with_entities = ray.get(extractor.process_documents.remote([document]))[0]
    print(f"Extracted {len(doc_with_entities.get('entities', []))} entities and {len(doc_with_entities.get('relationships', []))} relationships")
    
    # Step 2: Create chunks
    chunks = ray.get(chunker.chunk_documents.remote([document]))
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings for chunks
    chunks_with_embeddings = ray.get(embedder.embed_documents.remote(chunks))
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Step 4: Add to knowledge graph
    # Option 1: Standard approach
    # success = ray.get(graph_builder.add_article_with_chunks.remote(doc_with_entities, chunks_with_embeddings))
    
    # Option 2: GraphRAG approach
    success = ray.get(graph_builder.add_document_with_graphrag.remote(doc_with_entities))
    
    if success:
        print(f"Successfully added document to knowledge graph: {document['title']}")
    else:
        print(f"Failed to add document to knowledge graph: {document['title']}")
        
    return success

def test_search():
    """Test different search methods against the knowledge graph."""
    # Create search actor
    search = KnowledgeGraphSearch.remote()
    
    # Create embedding model for queries
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"))
    
    # Test queries
    queries = [
        "What new AI models were released?",
        "Who is Sam Altman?",
        "What companies are working on large language models?",
        "How do GPT-5 and Gemini Pro 2 compare?"
    ]
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        
        # Generate embedding for query
        embedding = model.encode(query, convert_to_tensor=False).tolist()
        
        # Test 1: Vector search
        print("\nVector search results:")
        results = ray.get(search.vector_search.remote(embedding, limit=2))
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            print(f"   Content: {result['content'][:100]}...")
        
        # Test 2: Hybrid search
        print("\nHybrid search results:")
        results = ray.get(search.hybrid_search.remote(query, embedding, limit=2))
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            print(f"   Title: {result.get('article_title', 'Unknown')}")
            print(f"   Content: {result['content'][:100]}...")
        
        # Test 3: Graph-enhanced search
        print("\nGraph-enhanced search results:")
        results = ray.get(search.graph_enhanced_search.remote(query, embedding, limit=2))
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            print(f"   Title: {result.get('article_title', 'Unknown')}")
            print(f"   Content: {result['content'][:100]}...")
            if "entity_context" in result and result["entity_context"]:
                print(f"   Entity context: {result['entity_context']}")

def main():
    """Main function to run the GraphRAG test."""
    initialize_ray()
    
    # Process test documents
    for doc in TEST_DOCS:
        process_document(doc)
    
    # Test search functionality
    test_search()
    
    print("\nGraphRAG test completed!")

if __name__ == "__main__":
    main()
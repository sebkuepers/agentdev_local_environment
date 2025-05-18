# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Ray-based LLM development template for building AI agents with local models. It provides an OpenAI-compatible API, integration with multiple agent frameworks (LangChain, LangGraph, CrewAI), a graph-based RAG system using Neo4j, and observability through Langfuse.

## Key Commands

### Development Environment

```bash
# Initialize environment configuration
just init-env

# Setup the environment with uv
just setup

# Download an LLM model
just download-model

# Start the complete stack (Ray, API, UI, Neo4j, Langfuse)
just start

# Check status of services
just status

# Stop all services
just stop
```

### Running Examples

```bash
# Run example implementations
just run-langchain   # Run LangChain agent example
just run-langgraph   # Run LangGraph workflow example
just run-crewai      # Run CrewAI team example
just run-graph-rag   # Run Graph RAG example
```

### Graph RAG System

```bash
# Start core services for Graph RAG
just start-rag-demo

# Initialize the knowledge graph
just kg-init

# Run the data pipeline once
just pipeline-run

# Start/stop continuous pipeline
just pipeline-start
just pipeline-stop
```

### Service Management

```bash
# Start/check individual services
just ray          # Start Ray cluster
just api          # Start OpenAI-compatible API
just ui           # Start OpenWebUI interface
just neo4j        # Start Neo4j database
just langfuse     # Start Langfuse observability

# View logs
just logs-ray     # View Ray dashboard
just logs-api     # View API server logs
just logs-ui      # View OpenWebUI logs
just logs-neo4j   # View Neo4j logs
just logs-pipeline # View pipeline logs
just logs-langfuse # View Langfuse logs
```

## Project Structure

The template follows a modular organization:

1. **Serving Layer** (`serve/` directory)
   - `basic.py` - Simple Ray Serve deployment
   - `openai_compatible.py` - Basic OpenAI compatibility layer
   - `openai_streaming.py` - OpenAI compatibility with streaming
   - `graph_rag_api.py` - RAG-enhanced OpenAI API

2. **Pipeline Components** (`pipeline/` directory)
   - `crawlers/` - Web crawlers for content ingestion
   - `processors/` - Text processing and chunking
   - `embedding/` - Embedding generation with local models
   - `graph/` - Neo4j knowledge graph operations
   - `pipeline.py` - Main pipeline orchestration
   - `scheduler.py` - Pipeline scheduling for continuous updates

3. **Examples** (`examples/` directory)
   - `langchain/` - LangChain-based agents
   - `langgraph/` - LangGraph workflows
   - `crewai/` - CrewAI multi-agent systems
   - `graph_rag/` - Graph-based RAG examples

4. **Configuration** (`config/` directory)
   - `ray_cluster.yaml` - Ray cluster configuration

## Configuration System

The project uses a `.env` file for configuration, which is loaded automatically by the dotenv library. Key configuration parameters:

```
# Model Settings
MODEL_DIR=~/ray-models
DEFAULT_MODEL=mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_TEMPERATURE=0.7
MAX_TOKENS=2000
CONTEXT_SIZE=4096
VERBOSE=true
GPU_LAYERS=-1

# Ray Cluster Settings
RAY_NUM_CPUS=12
RAY_NUM_GPUS=1
RAY_MEMORY=40000000000
RAY_DASHBOARD_HOST=0.0.0.0
RAY_PORT=6379

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
OPENWEBUI_PORT=3000
LANGFUSE_PORT=3021

# Neo4j Settings
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_BROWSER_PORT=7474

# Embedding Model Settings
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# Data Pipeline Settings
CRAWL_INTERVAL=3600  # Seconds between crawls
RSS_SOURCES=https://openai.com/blog/rss.xml,https://huggingface.co/blog/feed.xml
```

Python code loads these using `load_dotenv()` and accesses them with `os.getenv()`. The justfile uses `env_var_or_default()` to access them.

Commands for managing the configuration:

```bash
just init-env   # Create default .env file from .env.example
just reset-env  # Reset to defaults (backs up current .env first)
```

## Graph RAG System Architecture

The Graph RAG system has the following components:

1. **Data Ingestion** (`pipeline/crawlers/rss_crawler.py`)
   - RSS feed crawler for fetching articles from AI news sources
   - Uses feedparser and newspaper3k for content extraction

2. **Text Processing** (`pipeline/processors/chunker.py`)
   - Splits articles into chunks with overlap
   - Maintains metadata and relationships

3. **Embedding Generation** (`pipeline/embedding/local_embedder.py`)
   - Uses Sentence Transformers for generating embeddings
   - Distributed processing with Ray

4. **Knowledge Graph** (`pipeline/graph/`)
   - `schema.py` - Defines Neo4j schema with vector search indexes
   - `builder.py` - Creates and updates graph nodes and relationships
   - `search.py` - Provides semantic search capabilities

5. **Pipeline Orchestration** (`pipeline/pipeline.py`)
   - Coordinates the entire process using Ray
   - Handles errors and logging

6. **Example Usage** (`examples/graph_rag/simple_rag.py`)
   - Demonstrates RAG with the knowledge graph
   - Query embedding + hybrid search + LLM generation

## Code Patterns

When working with code in this repository, note these patterns:

1. **Model Initialization**:
   ```python
   # Load environment variables
   load_dotenv()
   
   # Get configuration
   MODEL_DIR = os.path.expanduser(os.getenv("MODEL_DIR", "~/ray-models"))
   DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
   MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
   
   # Build the model path
   model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
   
   # LlamaCpp initialization
   llm = LlamaCpp(
       model_path=model_path,
       temperature=MODEL_TEMPERATURE,
       max_tokens=MAX_TOKENS,
       n_ctx=CONTEXT_SIZE,
       callback_manager=callback_manager,
       n_gpu_layers=GPU_LAYERS,
       verbose=VERBOSE,
   )
   ```

2. **Ray Actor Pattern**:
   Most components are implemented as Ray actors for distributed processing:
   ```python
   @ray.remote
   class MyComponent:
       def __init__(self, config_param=None):
           # Initialize with config
           
       def process(self, data):
           # Process data
           return result
   
   # Usage:
   component = MyComponent.remote()
   result_ref = component.process.remote(data)
   result = ray.get(result_ref)
   ```

3. **Neo4j Integration**:
   ```python
   from neo4j import GraphDatabase
   
   # Connection
   driver = GraphDatabase.driver(uri, auth=(user, password))
   
   # Query execution
   with driver.session() as session:
       result = session.run("MATCH (n) RETURN n LIMIT 10")
       for record in result:
           # Process record
   ```

4. **Pipeline Orchestration**:
   ```python
   # Create actors
   crawler = Crawler.remote()
   processor = Processor.remote()
   embedder = Embedder.remote()
   graph = GraphBuilder.remote()
   
   # Pipeline execution
   data = ray.get(crawler.get_data.remote())
   processed = ray.get(processor.process.remote(data))
   with_embeddings = ray.get(embedder.embed.remote(processed))
   ray.get(graph.store.remote(with_embeddings))
   ```

## Development Notes

- Use uv instead of pip for package management
- The project relies on `just` for command orchestration - check `justfile` for available commands
- Models are stored in `~/ray-models/[model-name]/`
- When importing modules from packages, use the proper Python package paths, e.g., `from serve.basic import ...` or `from pipeline.graph.search import ...`
- When running examples, activate the virtual environment first or use the `just run-*` commands
- All configuration should use environment variables from `.env` file for consistency
- The Neo4j browser is available at http://localhost:7474 with credentials neo4j/password
- Pipeline logs are stored in `pipeline.log`
- Ray dashboard is available at http://localhost:8265
- OpenWebUI chat interface is available at http://localhost:3000
- Use the API directly at http://localhost:8000/v1
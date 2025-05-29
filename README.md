# Ray LLM Development Template

A comprehensive starter template for developing AI agents with local LLMs using Ray, LangChain, LangGraph, CrewAI, and Langfuse. Get up and running quickly with everything you need for local AI development!

## Overview

This template provides everything you need to set up a local development environment for building and deploying AI agents using your own local models. Perfect for developers who want full control over their AI stack! It includes:

- **Ray Distributed Computing**: Run models efficiently on your local machine
- **OpenAI-Compatible API**: Use standard OpenAI clients with your local models
- **Multiple Agent Frameworks**: LangChain, LangGraph, and CrewAI
- **Graph RAG Pipeline**: Build knowledge graphs from web content using Neo4j
- **Observability**: Langfuse integration for monitoring and debugging
- **Web UI**: Chat interface through OpenWebUI

## Quick Start

### 1. Setup

Clone this repository and run the setup script:

```bash
# Clone the repository
git clone https://github.com/sebkuepers/agentdev_local_environment.git
cd agentdev_local_environment

# Initialize environment configuration
just init-env

# Run the setup script (installs uv, creates venv, installs dependencies)
just setup
```

### 2. Download a Model

Download a local model to get started:

```bash
just download-model
```

### 3. Start the Stack

Start all services using tmux:

```bash
just start
```

Or start in background mode:

```bash
just start-bg
```

### 4. Access Services

Once everything is running, you can access these services:

- **Chat Interface**: http://localhost:3000 (OpenWebUI)
- **Ray Dashboard**: http://localhost:8265
- **Neo4j Browser**: http://localhost:7474
- **Langfuse Observability**: http://localhost:3021
- **OpenAI API**: http://localhost:8000/v1

## Configuration

The template uses a `.env` file for all configuration. When you run `just init-env`, a default configuration is created from [`.env.example`](.env.example). You can edit this file to customize:

- **Model settings**: Path, temperature, context size, etc.
- **Ray cluster settings**: CPU, GPU, and memory allocation
- **Server settings**: Host and port configuration
- **Neo4j settings**: Database connection parameters
- **Pipeline settings**: Crawl intervals, sources, chunk sizes
- **Langfuse settings**: Observability configuration

Key configuration options:

```env
# Model Settings
MODEL_DIR=~/ray-models
DEFAULT_MODEL=mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_TEMPERATURE=0.7
MAX_TOKENS=2000
CONTEXT_SIZE=4096

# Ray Cluster Settings
RAY_NUM_CPUS=12
RAY_NUM_GPUS=1

# Neo4j Settings
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# RAG Settings
USE_RAG=true
RAG_RESULTS_LIMIT=3

# Data Pipeline Settings
CRAWL_INTERVAL=3600  # Seconds between crawls
RSS_SOURCES=https://openai.com/blog/rss.xml,https://huggingface.co/blog/feed.xml
```

To reset your configuration to defaults:

```bash
just reset-env
```

## Graph RAG System

This template includes a complete graph-based RAG (Retrieval Augmented Generation) system that:

1. **Crawls content** from AI blogs and news sources
2. **Processes articles** into chunks optimized for retrieval
3. **Generates embeddings** using local models
4. **Builds a knowledge graph** in Neo4j with vector search capability
5. **Provides tools** for semantic search and question answering

### Setting up the Graph RAG System

```bash
# Start the core services for Graph RAG
just start-rag-demo

# Initialize the knowledge graph schema
just kg-init

# Run the data pipeline once to populate the graph
just pipeline-run

# Start the continuous data pipeline (runs in background)
just pipeline-start

# Try the Graph RAG example
just run-graph-rag
```

### ChatUI with Knowledge Graph Integration

The template includes an OpenAI-compatible API with knowledge graph integration that can be used with OpenWebUI:

```bash
# Start the chat interface with RAG capability
just start-rag-ui
```

This connects OpenWebUI to an API endpoint that:
1. Takes each user message
2. Searches the knowledge graph for relevant information
3. Adds the retrieved context to the prompt
4. Generates a response using the local LLM enhanced with context

**Features:**
- Seamless integration with standard chat interface
- Responses include citations from the knowledge graph
- All processing happens locally and privately
- Control RAG behavior through environment variables

### Managing the Knowledge Graph

```bash
# View the knowledge graph in Neo4j Browser
# http://localhost:7474 (neo4j/password)

# Stop the data pipeline
just pipeline-stop

# View pipeline logs
just logs-pipeline
```

## Run Examples

The repository includes example implementations for different agent frameworks:

```bash
# Run a simple LangChain agent
just run-langchain

# Run a LangGraph reasoning workflow
just run-langgraph

# Run a CrewAI research team
just run-crewai

# Run a Graph RAG question answering system
just run-graph-rag
```

## Management Commands

| Command | Description |
|---------|-------------|
| `just ray` | Start Ray cluster |
| `just api` | Start basic OpenAI-compatible API |
| `just api-rag` | Start OpenAI-compatible API with RAG |
| `just ui` | Start OpenWebUI chat interface |
| `just neo4j` | Start Neo4j database |
| `just langfuse` | Start Langfuse observability |
| `just kg-init` | Initialize knowledge graph schema |
| `just pipeline-run` | Run data pipeline once |
| `just pipeline-start` | Start continuous data pipeline |
| `just pipeline-stop` | Stop data pipeline |
| `just start-rag-ui` | Start UI with RAG capability |
| `just status` | Check service status |
| `just stop` | Stop all services |
| `just logs` | View available logs |
| `just init-env` | Create default configuration |
| `just reset-env` | Reset configuration to defaults |

## Directory Structure

```text
agentdev_local_environment/
├── .venv/                     # Created by setup.sh
├── config/                    # Configuration files
│   └── ray_cluster.yaml       # Ray cluster configuration
├── [examples/](examples/)                  # Example agent implementations
│   ├── [langchain/](examples/langchain/)             # LangChain examples
│   ├── [langgraph/](examples/langgraph/)             # LangGraph examples
│   ├── [crewai/](examples/crewai/)                # CrewAI examples
│   └── [graph_rag/](examples/graph_rag/)             # Graph RAG examples
├── pipeline/                  # Data pipeline components
│   ├── crawlers/              # Web crawlers
│   ├── processors/            # Text processing
│   ├── embedding/             # Embedding generation
│   └── graph/                 # Knowledge graph operations
├── langfuse-local/            # Langfuse Docker config
├── scripts/                   # Helper scripts
│   ├── [download_model.sh](scripts/download_model.sh)      # Model downloader
│   └── [setup.sh](scripts/setup.sh)               # Setup script
├── serve/                     # Model serving code
│   ├── basic.py               # Basic Ray Serve deployment
│   ├── openai_compatible.py   # OpenAI API (non-streaming)
│   ├── openai_streaming.py    # OpenAI API (with streaming)
│   └── graph_rag_api.py       # RAG-enhanced OpenAI API
├── [.env.example](.env.example)               # Example configuration file
├── [docker-compose.neo4j.yml](docker-compose.neo4j.yml)   # Neo4j Docker configuration 
├── .gitignore                 # Git ignore file
├── [justfile](justfile)                   # Command runner
├── [README.md](README.md)                  # Project documentation
└── [requirements.txt](requirements.txt)           # Python dependencies
```

## Requirements

- Python 3.9+
- Docker (for OpenWebUI, Neo4j, and Langfuse)
- Just command runner (`brew install just` on macOS)
- git
- wget or curl

## Additional Documentation

- [AGENTS.md](AGENTS.md) - Instructions for agents working on this repository
- [CLAUDE.md](CLAUDE.md) - Guidance for Claude Code when working with this codebase

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
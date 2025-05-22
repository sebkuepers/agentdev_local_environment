# AGENTS.md

This file summarizes instructions for agents working on this repository. It is based on the guidance originally written in `CLAUDE.md`.

## Overview

The project is a Ray-based development template for running local language models and building AI agents. It exposes an OpenAI-compatible API, integrates multiple agent frameworks (LangChain, LangGraph, CrewAI), includes a graph-based RAG pipeline backed by Neo4j, and provides observability via Langfuse.

## Common Commands

Use the `just` command runner for all operations.

### Environment Setup

```bash
just init-env      # create .env from example if missing
just setup         # create virtual environment and install dependencies using uv
just download-model # download a model
```

### Start/Stop Services

```bash
just start         # start Ray, API, UI, Neo4j and Langfuse in tmux
just status        # show service status
just stop          # stop all services
```

### Running Examples

```bash
just run-langchain   # LangChain agent example
just run-langgraph   # LangGraph workflow example
just run-crewai      # CrewAI team example
just run-graph-rag   # Graph RAG example
```

### Graph RAG Commands

```bash
just start-rag-demo  # start core services
just kg-init         # initialize Neo4j schema
just pipeline-run    # run the data pipeline once
just pipeline-start  # start continuous pipeline
just pipeline-stop   # stop pipeline
```

### Service Management

```bash
just ray          # start Ray cluster
just api          # start API server
just api-rag      # start API with RAG
just ui           # start OpenWebUI
just neo4j        # start Neo4j database
just langfuse     # start Langfuse dashboard
```

Logs can be viewed with `just logs-*` (e.g. `just logs-ray`, `just logs-api`).

## Project Structure

- `serve/` – Ray Serve deployments and API implementations
- `pipeline/` – data ingestion, processing, embedding, and graph code
- `examples/` – sample agents for LangChain, LangGraph, CrewAI and RAG
- `config/` – configuration files (e.g. `ray_cluster.yaml`)

## Configuration

All settings come from a `.env` file loaded with `dotenv`. Key variables include model paths, Ray cluster resources, server ports, Neo4j credentials, and pipeline options. Create or reset the file with `just init-env` or `just reset-env`.

## Code Patterns

When adding code, follow these patterns:

1. **Model initialization** using environment variables and LlamaCpp from `serve` modules.
2. **Ray actor pattern** for distributed components:
   ```python
   @ray.remote
   class MyComponent:
       def process(self, data):
           ...
   result = ray.get(MyComponent.remote().process.remote(data))
   ```
3. **Neo4j integration** via the official driver and sessions.
4. **Pipeline orchestration** that composes crawler, processor, embedder, and graph builder actors.

## Development Notes

- Use **uv** instead of pip for package management.
- `just` is the recommended interface for running scripts and services.
- Models are stored under `~/ray-models/[model-name]/`.
- Import modules using their package paths (e.g. `from serve.basic import ...`).
- Activate the virtual environment or use `just run-*` when running examples.
- Keep configuration in `.env` for consistency.
- Local dashboards:
  - Ray: <http://localhost:8265>
  - Neo4j Browser: <http://localhost:7474> (neo4j/password)
  - OpenWebUI: <http://localhost:3000>
  - Langfuse: <http://localhost:3021>



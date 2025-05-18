# AI Development Stack Commands

# Load environment variables from .env file
set dotenv-load := true

# Default environment variables
RAY_NUM_CPUS := env_var_or_default("RAY_NUM_CPUS", "12")
RAY_NUM_GPUS := env_var_or_default("RAY_NUM_GPUS", "1")
RAY_MEMORY := env_var_or_default("RAY_MEMORY", "40000000000")
RAY_DASHBOARD_HOST := env_var_or_default("RAY_DASHBOARD_HOST", "0.0.0.0")
RAY_PORT := env_var_or_default("RAY_PORT", "6379")
API_HOST := env_var_or_default("API_HOST", "0.0.0.0")
API_PORT := env_var_or_default("API_PORT", "8000")
OPENWEBUI_PORT := env_var_or_default("OPENWEBUI_PORT", "3000")
LANGFUSE_PORT := env_var_or_default("LANGFUSE_PORT", "3021")
OPENWEBUI_API_KEY := env_var_or_default("OPENWEBUI_API_KEY", "sk-111111111111111111111111111111111111111111111111")
DEFAULT_MODEL_NAME := env_var_or_default("DEFAULT_MODEL_NAME", "local-mistral-7b")
NEO4J_BROWSER_PORT := env_var_or_default("NEO4J_BROWSER_PORT", "7474")
NEO4J_PORT := env_var_or_default("NEO4J_PORT", "7687")
USE_RAG := env_var_or_default("USE_RAG", "true")

# List available commands
default:
    @just --list

# Setup the development environment
setup:
    #!/bin/bash
    echo "Setting up development environment..."
    chmod +x scripts/setup.sh
    chmod +x scripts/download_model.sh
    ./scripts/setup.sh

# Download a model
download-model:
    #!/bin/bash
    echo "Launching model downloader..."
    chmod +x scripts/download_model.sh
    ./scripts/download_model.sh

# Start the Ray cluster
ray:
    #!/bin/bash
    echo "Starting Ray cluster..."
    source .venv/bin/activate
    if ray status &>/dev/null; then
        echo "Ray is already running"
    else
        ray start --head --port={{RAY_PORT}} --dashboard-host={{RAY_DASHBOARD_HOST}} --num-cpus={{RAY_NUM_CPUS}} --num-gpus={{RAY_NUM_GPUS}} --memory={{RAY_MEMORY}}
        echo "✅ Ray cluster started successfully!"
    fi

# Deploy the OpenAI-compatible API
api:
    #!/bin/bash
    echo "Deploying OpenAI-compatible API..."
    source .venv/bin/activate
    # Check if Ray is running
    if ! ray status &>/dev/null; then
        echo "⚠️ Ray cluster is not running! Starting it now..."
        just ray
    fi
    
    # Shutdown any existing serve applications
    echo "Shutting down any existing applications..."
    serve shutdown || true
    
    # Deploy the API server
    echo "Starting API server..."
    serve run serve.openai_streaming:openai_app --host {{API_HOST}} --port {{API_PORT}}

# Deploy the OpenAI-compatible API with RAG capability
api-rag:
    #!/bin/bash
    echo "Deploying OpenAI-compatible API with RAG capability..."
    source .venv/bin/activate
    
    # Check if Ray is running
    if ! ray status &>/dev/null; then
        echo "⚠️ Ray cluster is not running! Starting it now..."
        just ray
    fi
    
    # Check if Neo4j is running
    if ! curl -s http://localhost:{{NEO4J_BROWSER_PORT}} &>/dev/null; then
        echo "⚠️ Neo4j is not running! Starting it now..."
        just neo4j
    fi
    
    # Shutdown any existing serve applications
    echo "Shutting down any existing applications..."
    serve shutdown || true
    
    # Deploy the API server with RAG capability
    echo "Starting RAG-enhanced API server..."
    export USE_RAG={{USE_RAG}}
    serve run serve.graph_rag_api:graph_rag_app --host {{API_HOST}} --port {{API_PORT}}

# Deploy the OpenAI-compatible API with RAG capability in background
api-rag-bg:
    #!/bin/bash
    source .venv/bin/activate
    if ! ray status &>/dev/null; then
        echo "⚠️ Ray cluster is not running! Starting it now..."
        just ray-bg
    fi
    
    # Check if Neo4j is running
    if ! curl -s http://localhost:{{NEO4J_BROWSER_PORT}} &>/dev/null; then
        echo "⚠️ Neo4j is not running! Starting it now..."
        just neo4j
    fi
    
    # Shutdown any existing serve applications
    serve shutdown &>/dev/null || true
    
    # Deploy the API server with RAG capability
    export USE_RAG={{USE_RAG}}
    nohup serve run serve.graph_rag_api:graph_rag_app --host {{API_HOST}} --port {{API_PORT}} > api.log 2>&1 &
    echo "✅ RAG-enhanced API server started in background"

# Start the OpenWebUI interface
ui:
    #!/bin/bash
    echo "Starting OpenWebUI container..."
    # Check if container exists and stop it
    docker stop openwebui &>/dev/null || true
    docker rm openwebui &>/dev/null || true
    
    # Start the container
    docker run -d \
      --name openwebui \
      --restart unless-stopped \
      -p {{OPENWEBUI_PORT}}:8080 \
      -e OPENAI_API_BASE_URL=http://host.docker.internal:{{API_PORT}}/v1 \
      -e HOST=0.0.0.0 \
      -e PORT=8080 \
      -e OPENAI_API_KEY={{OPENWEBUI_API_KEY}} \
      -e DEFAULT_MODELS={{DEFAULT_MODEL_NAME}} \
      -v ~/openwebui-data:/app/backend/data \
      --add-host host.docker.internal:host-gateway \
      ghcr.io/open-webui/open-webui:main
    
    echo "✅ OpenWebUI started at http://localhost:{{OPENWEBUI_PORT}}"

# Start Neo4j database
neo4j:
    #!/bin/bash
    echo "Starting Neo4j database..."
    # Make sure we have a .env file
    if [ ! -f .env ]; then
        just init-env
    fi
    
    # Start the container
    cd ~/dev/ray-cluster && docker-compose -f docker-compose.neo4j.yml up -d
    echo "✅ Neo4j started"
    echo "  Browser UI: http://localhost:{{NEO4J_BROWSER_PORT}}"
    echo "  Bolt URI: bolt://localhost:{{NEO4J_PORT}}"
    echo "  Default credentials: ${NEO4J_USER}/${NEO4J_PASSWORD}"

# Stop Neo4j database
neo4j-stop:
    #!/bin/bash
    echo "Stopping Neo4j database..."
    cd ~/dev/ray-cluster && docker-compose -f docker-compose.neo4j.yml down
    echo "✅ Neo4j stopped"

# Run the knowledge graph initialization
kg-init:
    #!/bin/bash
    echo "Initializing knowledge graph in Neo4j..."
    source .venv/bin/activate
    python -m pipeline.graph.schema
    echo "✅ Knowledge graph schema initialized"

# Run the data pipeline once
pipeline-run:
    #!/bin/bash
    echo "Running data pipeline..."
    source .venv/bin/activate
    python -m pipeline.pipeline
    echo "✅ Data pipeline run complete"

# Start the data pipeline scheduler
pipeline-start:
    #!/bin/bash
    echo "Starting data pipeline scheduler..."
    source .venv/bin/activate
    nohup python -m pipeline.scheduler > pipeline.log 2>&1 &
    echo "✅ Pipeline scheduler started in background"
    echo "  View logs: just logs-pipeline"

# Stop the data pipeline scheduler
pipeline-stop:
    #!/bin/bash
    echo "Stopping data pipeline scheduler..."
    pkill -f "python -m pipeline.scheduler" || echo "  ℹ️ No pipeline scheduler was running"
    echo "✅ Pipeline scheduler stopped"

# Run the graph RAG example
run-graph-rag:
    #!/bin/bash
    echo "Running graph RAG example..."
    source .venv/bin/activate
    python -m examples.graph_rag.simple_rag

# Start the AI observability dashboard (Langfuse)
langfuse:
    #!/bin/bash
    echo "Starting Langfuse observability stack..."
    # Create the langfuse directory if it doesn't exist
    mkdir -p langfuse-local
    
    # Create docker-compose file if it doesn't exist
    if [ ! -f langfuse-local/docker-compose.yml ]; then
        echo "Creating Langfuse docker-compose.yml..."
        cat > langfuse-local/docker-compose.yml << EOF
version: '3.8'
services:
  postgres:
    image: postgres:15
    restart: always
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - 5432:5432
  langfuse:
    image: langfuse/langfuse:latest
    restart: always
    depends_on:
      - postgres
    ports:
      - ${LANGFUSE_PORT}:3000
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - NEXTAUTH_SECRET=mysecret
      - SALT=mysalt
      - NEXTAUTH_URL=http://localhost:${LANGFUSE_PORT}
      - LANGFUSE_CLOUD_PROJECT_ID=""
      - LANGFUSE_CLOUD_PROJECT_SECRET=""
volumes:
  postgres_data:
EOF
    fi
    
    # Start the stack
    cd langfuse-local && docker-compose up -d
    echo "✅ Langfuse dashboard started at http://localhost:{{LANGFUSE_PORT}}"
    echo "   Default credentials: admin@langfuse.com / password"

# Run example LangChain agent
run-langchain:
    #!/bin/bash
    echo "Running LangChain agent example..."
    source .venv/bin/activate
    python -m examples.langchain.simple_agent

# Run example LangGraph workflow
run-langgraph:
    #!/bin/bash
    echo "Running LangGraph workflow example..."
    source .venv/bin/activate
    python -m examples.langgraph.reasoning_workflow

# Run example CrewAI team
run-crewai:
    #!/bin/bash
    echo "Running CrewAI research team example..."
    source .venv/bin/activate
    python -m examples.crewai.research_team

# Create a default .env file
init-env:
    #!/bin/bash
    if [ ! -f .env ]; then
        echo "Creating default .env file..."
        cp .env.example .env
        echo "✅ Created .env file from .env.example"
    else
        echo "⚠️ .env file already exists. To reset, run: just reset-env"
    fi

# Reset .env file to defaults
reset-env:
    #!/bin/bash
    if [ -f .env ]; then
        echo "Backing up existing .env to .env.backup..."
        cp .env .env.backup
    fi
    cp .env.example .env
    echo "✅ Reset .env file to defaults"

# Start everything for graph RAG demo
start-rag-demo: ray neo4j api-rag
    echo "✅ Core services for graph RAG demo are running"
    echo "  To initialize the knowledge graph: just kg-init"
    echo "  To run the data pipeline: just pipeline-run"
    echo "  To try the RAG example: just run-graph-rag"
    echo "  ChatUI with RAG: http://localhost:{{OPENWEBUI_PORT}} (after running 'just ui')"

# Start UI with RAG capabilities
start-rag-ui: ray-bg neo4j api-rag-bg ui
    echo "✅ RAG-enhanced ChatUI is running"
    echo "  Access at: http://localhost:{{OPENWEBUI_PORT}}"
    echo "  Knowledge graph browser: http://localhost:{{NEO4J_BROWSER_PORT}}"

# Start everything in tmux
start:
    #!/bin/bash
    echo "Starting complete AI stack in tmux..."
    
    # Install tmux if not installed
    if ! command -v tmux >/dev/null 2>&1; then
        if command -v brew >/dev/null 2>&1; then
            brew install tmux
        elif command -v apt >/dev/null 2>&1; then
            sudo apt install -y tmux
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y tmux
        else
            echo "⚠️ Please install tmux manually"
            exit 1
        fi
    fi
    
    # Make sure we have a .env file
    if [ ! -f .env ]; then
        just init-env
    fi
    
    # Kill existing session if it exists
    tmux kill-session -t ai-stack &>/dev/null || true
    
    # Create a new tmux session
    tmux new-session -d -s ai-stack
    
    # Start Ray in the first window
    tmux send-keys -t ai-stack "just ray" C-m
    tmux rename-window -t ai-stack "Ray Cluster"
    
    # Wait for Ray to start
    sleep 3
    
    # Create a new window for Neo4j
    tmux new-window -t ai-stack
    tmux send-keys -t ai-stack "just neo4j" C-m
    tmux rename-window -t ai-stack "Neo4j"
    
    # Wait for Neo4j to start
    sleep 3
    
    # Create a new window for the API
    tmux new-window -t ai-stack
    tmux send-keys -t ai-stack "just api-rag" C-m
    tmux rename-window -t ai-stack "API Server"
    
    # Wait for the API to start
    sleep 3
    
    # Create a new window for OpenWebUI
    tmux new-window -t ai-stack
    tmux send-keys -t ai-stack "just ui" C-m
    tmux rename-window -t ai-stack "OpenWebUI"
    
    # Create a new window for Langfuse
    tmux new-window -t ai-stack
    tmux send-keys -t ai-stack "just langfuse" C-m
    tmux rename-window -t ai-stack "Langfuse"
    
    # Return to first window
    tmux select-window -t ai-stack:0
    
    # Attach to the tmux session
    tmux attach-session -t ai-stack
    
    echo "AI stack started!"
    echo "Services available at:"
    echo "- OpenWebUI with RAG: http://localhost:{{OPENWEBUI_PORT}}"
    echo "- Ray Dashboard: http://localhost:8265"
    echo "- Neo4j Browser: http://localhost:{{NEO4J_BROWSER_PORT}}"
    echo "- Langfuse: http://localhost:{{LANGFUSE_PORT}}"
    echo "To detach from tmux: press Ctrl+B, then D"
    echo "To reattach: tmux attach-session -t ai-stack"

# Start everything (background mode - no tmux)
start-bg: ray-bg neo4j api-rag-bg ui langfuse
    echo "✅ All services started in background"

# Start Ray in background
ray-bg:
    #!/bin/bash
    source .venv/bin/activate
    if ray status &>/dev/null; then
        echo "Ray is already running"
    else
        nohup ray start --head --port={{RAY_PORT}} --dashboard-host={{RAY_DASHBOARD_HOST}} --num-cpus={{RAY_NUM_CPUS}} --num-gpus={{RAY_NUM_GPUS}} --memory={{RAY_MEMORY}} > ray.log 2>&1 &
        echo "✅ Ray cluster started in background"
    fi

# Start API in background 
api-bg:
    #!/bin/bash
    source .venv/bin/activate
    if ! ray status &>/dev/null; then
        echo "⚠️ Ray cluster is not running! Starting it now..."
        just ray-bg
    fi
    
    # Shutdown any existing serve applications
    serve shutdown &>/dev/null || true
    
    # Deploy the API server
    nohup serve run serve.openai_streaming:openai_app --host {{API_HOST}} --port {{API_PORT}} > api.log 2>&1 &
    echo "✅ API server started in background"

# Check status of all services
status:
    #!/bin/bash
    echo "Checking service status..."
    echo "Ray cluster:"
    if ray status &>/dev/null; then
        echo "  ✅ Running"
        echo "  Dashboard: http://localhost:8265"
    else
        echo "  ❌ Not running"
    fi
    
    echo "API server:"
    if curl -s http://localhost:{{API_PORT}}/v1/models &>/dev/null; then
        echo "  ✅ Running"
        echo "  Available models:"
        curl -s http://localhost:{{API_PORT}}/v1/models | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | sed 's/^/    - /'
    else
        echo "  ❌ Not running"
    fi
    
    echo "OpenWebUI:"
    if docker ps | grep -q openwebui; then
        echo "  ✅ Running (http://localhost:{{OPENWEBUI_PORT}})"
    else
        echo "  ❌ Not running"
    fi
    
    echo "Neo4j:"
    if docker ps | grep -q neo4j; then
        echo "  ✅ Running (http://localhost:{{NEO4J_BROWSER_PORT}})"
    else
        echo "  ❌ Not running"
    fi
    
    echo "Data Pipeline:"
    if pgrep -f "python -m pipeline.scheduler" > /dev/null; then
        echo "  ✅ Running"
    else
        echo "  ❌ Not running"
    fi
    
    echo "Langfuse:"
    if curl -s http://localhost:{{LANGFUSE_PORT}} &>/dev/null; then
        echo "  ✅ Running (http://localhost:{{LANGFUSE_PORT}})"
    else
        echo "  ❌ Not running"
    fi

# Stop all services
stop:
    #!/bin/bash
    echo "Stopping all services..."
    
    echo "Stopping Ray cluster..."
    ray stop &>/dev/null && echo "  ✅ Ray stopped" || echo "  ℹ️ Ray was not running"
    
    echo "Stopping OpenWebUI..."
    docker stop openwebui &>/dev/null && echo "  ✅ OpenWebUI stopped" || echo "  ℹ️ OpenWebUI was not running"
    
    echo "Stopping Neo4j..."
    cd ~/dev/ray-cluster && docker-compose -f docker-compose.neo4j.yml down && echo "  ✅ Neo4j stopped" || echo "  ℹ️ Neo4j was not running"
    
    echo "Stopping Data Pipeline..."
    pkill -f "python -m pipeline.scheduler" &>/dev/null && echo "  ✅ Pipeline stopped" || echo "  ℹ️ Pipeline was not running" 
    
    echo "Stopping Langfuse..."
    if [ -f langfuse-local/docker-compose.yml ]; then
        cd langfuse-local && docker-compose down && echo "  ✅ Langfuse stopped" || echo "  ℹ️ Langfuse was not running"
    else
        echo "  ℹ️ Langfuse was not running"
    fi
    
    echo "Stopping any tmux sessions..."
    tmux kill-session -t ai-stack &>/dev/null && echo "  ✅ Tmux session stopped" || echo "  ℹ️ No tmux session was running"
    
    echo "✅ All services stopped"

# Show logs
logs:
    #!/bin/bash
    echo "Available logs:"
    echo "1. Ray dashboard (ray): http://localhost:8265"
    echo "2. API server logs (api): api.log"
    echo "3. OpenWebUI logs (ui): docker logs openwebui"
    echo "4. Neo4j logs (neo4j): docker logs ray-cluster_neo4j_1"
    echo "5. Pipeline logs (pipeline): pipeline.log" 
    echo "6. Langfuse logs (langfuse): docker logs langfuse-local_langfuse_1"
    echo ""
    echo "To view a specific log, run:"
    echo "  just logs-ray    - Open Ray dashboard"
    echo "  just logs-api    - View API logs"
    echo "  just logs-ui     - View OpenWebUI logs"
    echo "  just logs-neo4j  - View Neo4j logs"
    echo "  just logs-pipeline - View pipeline logs"
    echo "  just logs-langfuse - View Langfuse logs"

# Show Ray logs
logs-ray:
    open http://localhost:8265

# Show API logs
logs-api:
    tail -f api.log

# Show OpenWebUI logs
logs-ui:
    docker logs -f openwebui

# Show Neo4j logs  
logs-neo4j:
    docker logs -f ray-cluster_neo4j_1

# Show Pipeline logs
logs-pipeline:
    tail -f pipeline.log

# Show Langfuse logs
logs-langfuse:
    docker logs -f langfuse-local_langfuse_1
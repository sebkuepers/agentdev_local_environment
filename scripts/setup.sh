#!/bin/bash
set -e

# Define colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up AI Development Environment${NC}"
echo -e "${YELLOW}This script will:${NC}"
echo "1. Install uv if not already installed"
echo "2. Create a virtual environment with uv"
echo "3. Install all required dependencies"
echo "4. Set up model directories"
echo -e "\n"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to the current PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}uv installed successfully!${NC}"
else
    echo -e "${GREEN}uv already installed.${NC}"
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using uv
echo -e "\n${YELLOW}Installing dependencies...${NC}"
uv pip install -r requirements.txt

# Create model directories
echo -e "\n${YELLOW}Setting up model directories...${NC}"
MODEL_DIR="$HOME/ray-models/mistral-7b-instruct"
mkdir -p "$MODEL_DIR"

echo -e "\n${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Download a model: just download-model"
echo "3. Start the Ray cluster: just ray"
echo -e "\n${GREEN}Happy coding!${NC}"
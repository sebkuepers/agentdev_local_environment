#!/bin/bash
set -e

# Define colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MODEL_DIR="$HOME/ray-models"
DEFAULT_MODEL="mistral-7b-instruct"

show_models() {
    echo -e "${YELLOW}Available models to download:${NC}"
    echo "1) Mistral 7B Instruct v0.2 (GGUF) [4-bit quantized, ~4GB]"
    echo "2) Mistral 7B Instruct v0.2 (GGUF) [5-bit quantized, ~5GB]"
    echo "3) Llama 3 8B Instruct (GGUF) [4-bit quantized, ~4.5GB]"
    echo "4) Phi-3 Mini (GGUF) [4-bit quantized, ~2GB]"
    echo "5) CodeLlama 7B Instruct (GGUF) [4-bit quantized, ~4GB]"
    echo "c) Custom model URL (GGUF format)"
}

download_model() {
    local model_name=$1
    local model_url=$2
    local model_file=$3
    
    # Create directory
    local model_path="$MODEL_DIR/$model_name"
    mkdir -p "$model_path"
    
    echo -e "${YELLOW}Downloading $model_name...${NC}"
    echo "This may take some time depending on your internet connection."
    
    # Download with curl and show progress
    curl -L "$model_url" --output "$model_path/$model_file" -#
    
    echo -e "${GREEN}Model downloaded successfully to:${NC}"
    echo "$model_path/$model_file"
    
    # Update config for the model
    update_model_config "$model_name/$model_file"
}

update_model_config() {
    local model_path=$1
    
    # Update the model paths in Python files
    for file in serve/*.py examples/*/*.py; do
        if grep -q "ray-models/.*/.*\.gguf" "$file"; then
            sed -i.bak "s|ray-models/.*/.*\.gguf|ray-models/$model_path|g" "$file"
            rm -f "$file.bak"
            echo "Updated model path in $file"
        fi
    done
    
    echo -e "${GREEN}Configuration updated to use the new model.${NC}"
}

# Main script
echo -e "${BLUE}AI Model Downloader${NC}"
echo "This script will download the selected model to $MODEL_DIR"
show_models
echo ""

read -p "Enter your choice (1-5 or c): " choice

case $choice in
    1)
        download_model "mistral-7b-instruct" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ;;
    2)
        download_model "mistral-7b-instruct" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf" "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
        ;;
    3)
        download_model "llama3-8b-instruct" "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf" "llama-3-8b-instruct.Q4_K_M.gguf"
        ;;
    4)
        download_model "phi3-mini" "https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf" "phi-3-mini-4k-instruct.Q4_K_M.gguf"
        ;;
    5)
        download_model "codellama-7b-instruct" "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf" "codellama-7b-instruct.Q4_K_M.gguf"
        ;;
    c)
        read -p "Enter the model name (e.g., 'custom-model'): " custom_name
        read -p "Enter the URL to download from: " custom_url
        read -p "Enter the filename to save as: " custom_filename
        download_model "$custom_name" "$custom_url" "$custom_filename"
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC} Now you can start the Ray cluster with 'just ray'"
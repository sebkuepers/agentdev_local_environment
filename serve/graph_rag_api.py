#!/usr/bin/env python3
"""
Graph RAG API - OpenAI-Compatible API with Knowledge Graph Integration

This module provides an OpenAI-compatible API that leverages the knowledge graph
for enhanced responses, creating a hybrid model service that combines local LLM inference
with context from the knowledge graph.
"""

import os
import ray
from ray import serve
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
import logging
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Import knowledge graph components
from pipeline.graph.search import KnowledgeGraphSearch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
MODEL_DIR = os.path.expanduser(os.getenv("MODEL_DIR", "~/ray-models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "local-mistral-7b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"

# Create a token collector for streaming
class SimpleTokenCollector(BaseCallbackHandler):
    """Callback handler that simply collects tokens for streaming."""
    
    def __init__(self):
        super().__init__()
        self.tokens: List[str] = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect tokens as they're generated."""
        self.tokens.append(token)

# Initialize FastAPI
app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class GraphRAGAPI:
    """An OpenAI-compatible API that integrates knowledge graph context."""
    
    def __init__(self):
        """Initialize the GraphRAG API."""
        # Build the model path
        self.model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
        
        # Load embedding model for queries
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            logger.info(f"Embedding model loaded on {device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
        
        # For regular (non-streaming) requests
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=GPU_LAYERS,
            verbose=VERBOSE,
        )
        
        # Model info
        self.model_name = DEFAULT_MODEL_NAME
        
        # Initialize knowledge graph search
        if USE_RAG:
            logger.info("Initializing knowledge graph search...")
            try:
                self.kg_search = KnowledgeGraphSearch.remote()
                logger.info("Knowledge graph search initialized!")
            except Exception as e:
                logger.error(f"Failed to initialize knowledge graph search: {e}")
                self.kg_search = None
        else:
            self.kg_search = None
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for a query."""
        if self.embedding_model is None:
            return None
            
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def retrieve_context(self, query: str, limit: int = 3) -> str:
        """Retrieve context from knowledge graph."""
        if self.kg_search is None:
            return ""
            
        try:
            # Generate query embedding
            embedding = self.get_embeddings(query)
            
            if embedding is None:
                return ""
            
            # Perform hybrid search
            results = await ray.get(self.kg_search.hybrid_search.remote(
                query=query,
                embedding=embedding,
                limit=limit,
                vector_weight=0.7
            ))
            
            if not results:
                return ""
            
            # Format retrieved chunks
            context_parts = []
            for i, chunk in enumerate(results):
                source_info = f"Source: {chunk.get('source', 'Unknown source')}"
                article_title = chunk.get('article_title', 'Untitled')
                article_url = chunk.get('article_url', '')
                
                # Format the chunk with metadata
                chunk_text = f"[{i+1}] From article '{article_title}' ({source_info}):\n{chunk['content']}\n"
                context_parts.append(chunk_text)
                
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
    
    @app.get("/v1/models")
    async def list_models(self):
        """List available models (OpenAI compatibility)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local-user",
                }
            ]
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        """Handle chat completions with optional RAG context."""
        data = await request.json()
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        
        # Format messages for LlamaCpp
        prompt = ""
        
        # Extract the last user message for context retrieval
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        
        # Retrieve context if RAG is enabled and we have a user message
        context = ""
        if USE_RAG and last_user_message:
            context = await self.retrieve_context(last_user_message)
            logger.info(f"Retrieved context: {len(context)} chars")
        
        # Build the prompt with system message and context
        # First find if there's a system message
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break
        
        # If we have context, add it to system message or create one
        if context:
            if system_content:
                system_content = f"{system_content}\n\nHere is some relevant information that might help you:\n{context}"
            else:
                system_content = f"You are a helpful assistant. Here is some relevant information that might help you:\n{context}"
        
        # Build the full prompt
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Replace system message with our augmented one if we have context
            if role == "system":
                content = system_content if context else content
                
            if role == "system":
                prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                if prompt:
                    prompt += f"{content} [/INST]"
                else:
                    prompt += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} </s><s>[INST] "
        
        # Handle streaming response
        if stream:
            return StreamingResponse(
                self._generate_streaming_response(prompt, data),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        completion = self.llm.invoke(prompt)
        
        # Format as OpenAI response
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(completion) // 4,
                "total_tokens": (len(prompt) + len(completion)) // 4
            }
        }
        
        return response
    
    async def _generate_streaming_response(self, prompt: str, data: Dict[str, Any]):
        """Generate a streaming response in the OpenAI format."""
        # Create streaming handler
        token_collector = SimpleTokenCollector()
        
        # For streaming we need a separate LlamaCpp instance with our handler
        stream_llm = LlamaCpp(
            model_path=self.model_path,
            temperature=data.get("temperature", MODEL_TEMPERATURE),
            max_tokens=data.get("max_tokens", MAX_TOKENS),
            n_ctx=CONTEXT_SIZE,
            callback_manager=CallbackManager([token_collector]),
            n_gpu_layers=GPU_LAYERS,
            verbose=False,
        )
        
        # Start the completion in a separate thread to avoid blocking
        generation_task = asyncio.create_task(
            asyncio.to_thread(stream_llm.invoke, prompt)
        )
        
        # Send response ID and role in the first chunk
        response_id = f"chatcmpl-{int(time.time())}"
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        
        # Buffer for smaller chunks
        buffer = ""
        token_count = 0
        
        # Monitor tokens being generated
        try:
            while not generation_task.done() or len(token_collector.tokens) > token_count:
                # Check for new tokens
                while token_count < len(token_collector.tokens):
                    token = token_collector.tokens[token_count]
                    buffer += token
                    token_count += 1
                    
                    # Send tokens as chunks (can adjust buffer size if needed)
                    if len(buffer) >= 4 or (token_count == len(token_collector.tokens) and generation_task.done()):
                        content_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": buffer
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n"
                        buffer = ""
                
                # Small delay to avoid tight polling
                await asyncio.sleep(0.01)
            
            # Final chunk with finish_reason
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # End of stream
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            # Send error in the stream format
            error_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": f"\n\nError during generation: {str(e)}"
                    },
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

# Deploy the service
graph_rag_app = GraphRAGAPI.bind()
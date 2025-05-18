import os
from ray import serve
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import json
import time
import asyncio
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
MODEL_DIR = os.path.expanduser(os.getenv("MODEL_DIR", "~/ray-models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "local-mistral-7b")

# Create a simpler token collector that doesn't try to override properties
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
class OpenAICompatibleLLM:
    def __init__(self):
        # Build the model path
        self.model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
        
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
    
    # OpenAI compatibility endpoints
    @app.get("/v1/models")
    async def list_models(self):
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
        data = await request.json()
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        
        # Format messages for LlamaCpp
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
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
            print(f"Streaming error: {str(e)}")
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
openai_app = OpenAICompatibleLLM.bind()
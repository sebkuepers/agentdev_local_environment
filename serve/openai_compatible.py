# Create a file named openai_compatible_ray.py
import os
import ray
import time
from ray import serve
from fastapi import FastAPI, Request, Response
import json
import asyncio
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize Ray and FastAPI
ray.init(address="auto")
app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class OpenAICompatibleLLM:
    def __init__(self):
        # Use absolute path for model
        home_dir = os.path.expanduser("~")
        self.model_path = os.path.join(home_dir, "ray-models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        
        # Initialize model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=0.7,
            max_tokens=2000,
            n_ctx=4096,
            callback_manager=callback_manager,
            n_gpu_layers=-1,
            verbose=True,
        )
        
        # Model info
        self.model_name = "local-mistral-7b"
    
    # OpenAI compatibility endpoints
    @app.get("/v1/models")
    async def list_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "local-user",
                }
            ]
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        data = await request.json()
        messages = data.get("messages", [])
        
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
        
        # Get completion
        completion = self.llm(prompt)
        
        # Format as OpenAI response
        response = {
            "id": f"chatcmpl-{int(time.time())}", # Use timestamp instead of job_id
            "object": "chat.completion",
            "created": int(time.time()),  # Use current time
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

# Deploy the service
openai_app = OpenAICompatibleLLM.bind()
serve.run(openai_app)
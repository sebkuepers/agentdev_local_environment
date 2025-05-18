import os
import ray
from ray import serve
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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

# Connect to Ray cluster
ray.init(address="auto")

@serve.deployment()
class LLMModel:
    def __init__(self):
        # Build the model path
        model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
        
        # Print the full path for debugging
        print(f"Looking for model at: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        # Initialize callbacks
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initialize model
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n_ctx=CONTEXT_SIZE,
            callback_manager=callback_manager,
            n_gpu_layers=GPU_LAYERS,
            verbose=VERBOSE,
        )
    
    async def __call__(self, request):
        prompt = await request.json()
        response = self.llm(prompt["text"])
        return {"response": response}

# Deploy the model
app = LLMModel.bind()
serve.run(app, route_prefix="/llm")
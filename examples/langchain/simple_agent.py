#!/usr/bin/env python3
"""
LangChain Simple Agent Example

This example demonstrates a basic LangChain agent using a local LLM served via Ray.
It uses a straightforward prompt -> LLM -> output chain pattern.
"""

import os
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Langfuse for observability (optional)
langfuse_handler = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    from langfuse.api.client import Langfuse
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_handler = LangfuseCallbackHandler()

# Initialize the LLM with LlamaCpp
def get_llm():
    """Initialize the local LLM with LlamaCpp."""
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, "ray-models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    if langfuse_handler:
        callback_manager.add_handler(langfuse_handler)
    
    return LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2000,
        n_ctx=4096,
        callback_manager=callback_manager,
        n_gpu_layers=-1,
        verbose=True,
    )

# Define the agent prompt
template = """You are a helpful AI assistant. Answer the user's question based on your knowledge.

User question: {question}

Your answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the chain
def create_chain():
    """Create a simple LangChain chain."""
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain

# Execute the agent
def run_agent(question: str) -> str:
    """Run the agent on a given question."""
    chain = create_chain()
    result = chain.invoke({"question": question})
    return result

if __name__ == "__main__":
    # Example usage
    questions = [
        "What are the benefits of using Ray for distributed computing?",
        "How does LangChain help with building AI applications?",
        "What are the advantages of using local LLMs instead of cloud APIs?"
    ]
    
    print("🤖 LangChain Simple Agent Example\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}\n")
        answer = run_agent(question)
        print(f"\n✅ Answer: {answer}\n")
        print("-" * 80)
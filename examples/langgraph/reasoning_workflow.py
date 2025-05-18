#!/usr/bin/env python3
"""
LangGraph Reasoning Workflow Example

This example demonstrates a multi-step reasoning workflow using LangGraph.
It implements a thinking → answering process for more reliable responses.
"""

import os
from typing import Dict, List, Any
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
def get_llm():
    """Initialize the local LLM with LlamaCpp."""
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, "ray-models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    return LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2000,
        n_ctx=4096,
        callback_manager=callback_manager,
        n_gpu_layers=-1,
        verbose=True,
    )

# Define the nodes in our graph
def thinking(state: Dict) -> Dict:
    """Think about the question before answering."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """You need to think step by step about this question:
        
        Question: {question}
        
        What are your thoughts about how to answer this question?
        Think about the context, relevant knowledge, and any considerations.
        Provide 3-5 bullet points of relevant thoughts."""
    )
    
    chain = prompt | llm | StrOutputParser()
    thoughts = chain.invoke({"question": state["question"]})
    
    return {"thoughts": state.get("thoughts", []) + [thoughts]}

def answering(state: Dict) -> Dict:
    """Answer the question based on thoughts."""
    llm = get_llm()
    
    thoughts = state.get("thoughts", [])
    thoughts_text = "\n".join(thoughts)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant that provides accurate information.
        
        Question: {question}
        
        You've thought about this and have these considerations:
        {thoughts}
        
        Based on your thoughts, provide a comprehensive answer to the question."""
    )
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": state["question"], 
        "thoughts": thoughts_text
    })
    
    return {"answer": answer}

# Build the graph
def build_graph():
    """Build and compile the LangGraph workflow."""
    # Initialize the graph
    workflow = StateGraph({"question": str, "thoughts": list, "answer": str})
    
    # Add nodes
    workflow.add_node("thinking", thinking)
    workflow.add_node("answering", answering)
    
    # Add edges
    workflow.add_edge("thinking", "answering")
    workflow.add_edge("answering", END)
    
    # Set entrypoint
    workflow.set_entry_point("thinking")
    
    # Compile the graph
    return workflow.compile()

# Function to run the reasoning workflow
def run_workflow(question: str) -> Dict[str, Any]:
    """Run the reasoning workflow on a given question."""
    graph = build_graph()
    result = graph.invoke({"question": question, "thoughts": [], "answer": ""})
    return result

if __name__ == "__main__":
    # Example usage
    questions = [
        "What are the key differences between LangChain and LangGraph?",
        "How does multi-step reasoning help with complex questions?",
        "What are the advantages of a workflow-based approach to AI reasoning?"
    ]
    
    print("🤖 LangGraph Reasoning Workflow Example\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}\n")
        
        print("🧠 Thinking...")
        result = run_workflow(question)
        
        print("\n💭 Thoughts:")
        print(result["thoughts"][0])
        
        print("\n✅ Final Answer:")
        print(result["answer"])
        print("\n" + "-" * 80)
#!/usr/bin/env python3
"""
CrewAI Research Team Example

This example demonstrates a multi-agent system using CrewAI.
It implements a research team with specialized roles:
- Research Analyst: Gathers information
- Content Writer: Creates content based on research
- Content Reviewer: Reviews and improves the content
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

# Define the CrewAI setup
llm = get_llm()

# Create the agents
researcher = Agent(
    role="Research Analyst",
    goal="Conduct detailed research on topics to gather accurate information",
    backstory="You are an expert research analyst with a talent for finding and synthesizing information.",
    verbose=True,
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging and informative content based on research",
    backstory="You are a skilled writer who can transform research into compelling content.",
    verbose=True,
    llm=llm,
)

reviewer = Agent(
    role="Content Reviewer",
    goal="Ensure content is accurate, informative, and well-structured",
    backstory="You have a keen eye for detail and a commitment to quality content.",
    verbose=True,
    llm=llm,
)

# Function to run the crew
def run_research_team(topic: str) -> str:
    """Run the research team on a given topic."""
    # Create tasks
    research_task = Task(
        description=f"Research the topic: {topic}. Gather key information, facts, statistics, and insights.",
        agent=researcher,
        expected_output="Comprehensive research notes on the topic",
    )
    
    writing_task = Task(
        description="Using the research, create informative and engaging content on the topic.",
        agent=writer,
        expected_output="Well-written article on the topic",
        context=[research_task],
    )
    
    review_task = Task(
        description="Review the content for accuracy, clarity, and engagement. Make any necessary improvements.",
        agent=reviewer,
        expected_output="Final polished content ready for publication",
        context=[writing_task],
    )
    
    # Create the crew
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        verbose=True,
        process=Process.sequential,
    )
    
    # Run the crew
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    # Example usage
    topics = [
        "The benefits of distributed computing with Ray",
        "How AI agents can collaborate to solve complex problems",
        "The future of local LLM deployment for businesses"
    ]
    
    print("🤖 CrewAI Research Team Example\n")
    
    # Select a topic
    print("Available research topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    try:
        choice = int(input("\nSelect a topic (1-3): "))
        if choice < 1 or choice > len(topics):
            raise ValueError
        selected_topic = topics[choice-1]
    except (ValueError, IndexError):
        print("Invalid choice, using default topic.")
        selected_topic = topics[0]
    
    print(f"\n📊 Starting research on: {selected_topic}")
    print("\n" + "=" * 80 + "\n")
    
    result = run_research_team(selected_topic)
    
    print("\n" + "=" * 80)
    print("\n✅ Final Result:\n")
    print(result)
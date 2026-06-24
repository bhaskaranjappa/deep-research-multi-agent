import os
import json
from typing import Annotated, Sequence, TypedDict
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tools import native_web_search
# Import precision retry utilities
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class QueryPlan(BaseModel):
    queries: list[str] = Field(description="List of 3-4 distinct search queries optimized for web search engines.")

# Structured schema for the Critic's decision making
class EvaluationResult(BaseModel):
    approved: bool = Field(description="True if the research summary fully covers the user request, False if it needs more detail.")
    critique: str = Field(description="Feedback on what information is missing or needs further lookup.")
    new_queries: list[str] = Field(description="If not approved, list 1-2 new specific search queries to run.")

class ResearchState(TypedDict):
    messages: Annotated[Sequence[dict], add_messages]
    search_queries: list[str]
    current_research: str
    loop_count: int  # Added to prevent infinite routing loops

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Production-grade exponential backoff wrapper
@retry(
    stop=stop_after_attempt(4), # Try 4 times max before giving up
    wait=wait_exponential(multiplier=2, min=2, max=10), # Wait 2s, 4s, 8s...
    reraise=True # If all attempts fail, pass the error gracefully to the UI
)
def call_gemini_with_retry(contents, config):
    """
    Executes a structured generative call against Gemini 2.5 Flash, 
    automatically backing off if hit by upstream 503 network strain.
    """
    return client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=contents, 
        config=config
    )

def planner_node(state: ResearchState):
    messages = state.get("messages", [])
    task_content = messages[-1].content if messages else "No task."
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QueryPlan,
        system_instruction="You are a research planning engine.",
        temperature=0.1
    )
    # Patched to leverage robust tenacity structure
    response = call_gemini_with_retry(f"Plan research for: {task_content}", config)
    data = json.loads(response.text)
    
    return {
        "search_queries": data.get("queries", []),
        "messages": [{"role": "assistant", "content": f"Initial plan generated."}],
        "loop_count": 0
    }

def researcher_node(state: ResearchState):
    queries = state.get("search_queries", [])
    scraped_context_chunks = []
    
    for query in queries:
        raw_results = native_web_search(query)
        scraped_context_chunks.append(f"### Query: {query}\nResults:\n{raw_results}")
    
    full_scraped_context = "\n\n".join(scraped_context_chunks)
    existing_research = state.get("current_research", "")
    
    config = types.GenerateContentConfig(
        system_instruction="You are a Senior AI Researcher. Compile your findings into a comprehensive technical doc.",
        temperature=0.2
    )
    # Patched to leverage robust tenacity structure
    response = call_gemini_with_retry(
        f"Existing Document Draft:\n{existing_research}\n\nNew context to integrate:\n{full_scraped_context}", 
        config
    )
    
    return {
        "messages": [{"role": "assistant", "content": "Researcher updated findings."}],
        "current_research": response.text
    }

def critic_node(state: ResearchState):
    messages = state.get("messages", [])
    original_task = messages[0].content if messages else "No task."
    current_research = state.get("current_research", "")
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EvaluationResult,
        system_instruction="You are a meticulous content critic. Evaluate if the research document hits all constraints of the task.",
        temperature=0.1
    )
    
    prompt = f"Original Task: {original_task}\n\nCurrent Draft:\n{current_research}"
    # Patched to leverage robust tenacity structure
    response = call_gemini_with_retry(prompt, config)
    eval_data = json.loads(response.text)
    
    return {
        "search_queries": eval_data.get("new_queries", []),
        "loop_count": state.get("loop_count", 0) + 1,
        "messages": [{"role": "assistant", "content": f"Critique: {eval_data.get('critique')}"}]
    }

def should_continue(state: ResearchState):
    if state.get("loop_count", 0) >= 2:
        print("--- HARD TERMINATION: Loop limit reached ---")
        return END
        
    messages = state.get("messages", [])
    last_message = messages[-1].content
    if "Critique:" in last_message and len(state.get("search_queries", [])) > 0:
        print("--- LOOPING BACK: Critic requested deeper data collection ---")
        return "researcher"
        
    print("--- SUCCESS: Content approved by Critic ---")
    return END

# Re-map graph topology setup
workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "researcher": "researcher",
        END: END
    }
)

research_graph = workflow.compile()
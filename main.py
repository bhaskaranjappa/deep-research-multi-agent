import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import research_graph

app = FastAPI(title="Deep Research State Machine")

class ResearchRequest(BaseModel):
    task: str

@app.get("/")
def health():
    return {"status": "active", "node": "langgraph_researcher_v1"}

@app.post("/research")
def trigger_research(payload: ResearchRequest):
    try:
        # Executing the LangGraph state machine synchronously for testing
        initial_state = {
            "messages": [{"role": "user", "content": payload.task}],
            "search_queries": [],
            "current_research": ""
        }
        
        output = research_graph.invoke(initial_state)
        
        # Pulling the updated state history out of the compiled graph execution run
        return {
            "final_research": output.get("current_research"),
            "execution_history": output.get("messages")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
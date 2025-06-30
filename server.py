import threading
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Import your functions from llm.py
from llm import load_llm, load_indices_tools, load_tools, query_agent, load_agent, dummy_query_agent

# Initialize LLM, load indices and tools
llm = load_llm()
print("Loaded LLM")
index_set = load_indices_tools()
print("Loaded indices")
tools = load_tools(llm, index_set)
print("Loaded tools")
agent = load_agent(llm, tools)
print("Loaded agent")
dummy_query_agent(llm, tools, agent, input="what is cs")
print("Completed dummy query 1")
dummy_query_agent(llm, tools, agent, input="what is math")
print("Completed dummy query 2")

# Create the FastAPI application
app = FastAPI()

# Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the query agent API"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Process the query using your query_agent function
    response = query_agent(llm, tools, agent, request.query)
    return {"response": response.response}

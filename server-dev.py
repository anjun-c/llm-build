import threading
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Import your functions from llm-dev.py
import importlib.util
import sys

# Load the llm-dev.py module
spec = importlib.util.spec_from_file_location("llm_dev", "llm-dev.py")
llm_dev = importlib.util.module_from_spec(spec)
sys.modules["llm_dev"] = llm_dev
spec.loader.exec_module(llm_dev)

# Import the functions
load_llm = llm_dev.load_llm
load_indices_tools = llm_dev.load_indices_tools
load_tools = llm_dev.load_tools
query_agent = llm_dev.query_agent
load_agent = llm_dev.load_agent
dummy_query_agent = llm_dev.dummy_query_agent

# Initialize LLM, load indices and tools
print("Initializing LangChain RAG system...")
llm = load_llm()
print("✓ Loaded LLM")

vector_stores = load_indices_tools()
print("✓ Loaded vector stores")

tools = load_tools(llm, vector_stores)
print("✓ Loaded tools")

agent = load_agent(llm, tools)
print("✓ Loaded agent")

# Test with dummy queries (only if Ollama is running)
try:
    dummy_query_agent(agent, "what is cs")
    print("✓ Completed dummy query 1")
    dummy_query_agent(agent, "what is math")
    print("✓ Completed dummy query 2")
except Exception as e:
    print(f"⚠️ Dummy queries failed (Ollama may not be running): {e}")
    print("✓ Server will still work for API requests when Ollama is available")

# Create the FastAPI application
app = FastAPI(title="LangChain RAG API", description="RAG system with enhanced source tracking")

# Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the LangChain RAG API with Enhanced Source Tracking",
        "features": [
            "Multi-subject knowledge base (Biology, Chemistry, Physics, Mathematics, Computer Science)",
            "Enhanced source tracking with file names and sections",
            "LangChain-based RAG implementation",
            "Structured response format with citations"
        ],
        "endpoints": {
            "/query": "POST - Submit a query to the RAG system",
            "/health": "GET - Check system health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "llm": "loaded",
            "vector_stores": f"{len(vector_stores)} subjects loaded",
            "tools": f"{len(tools)} RAG tools available",
            "agent": "initialized"
        }
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with enhanced source tracking.
    
    Returns answers with detailed source information including:
    - Document names (file names)
    - Section information (chunk positions)
    - Subject classifications
    """
    try:
        # Process the query using your query_agent function
        response = query_agent(agent, request.query)
        
        # Extract the response content
        response_text = response.get('output', 'No response generated')
        
        return {
            "query": request.query,
            "response": response_text,
            "status": "success",
            "note": "Response includes source citations with file names and sections"
        }
    except Exception as e:
        return {
            "query": request.query,
            "response": f"Error processing query: {str(e)}",
            "status": "error",
            "note": "Make sure Ollama is running with the tinyllama model"
        }

@app.get("/subjects")
async def list_subjects():
    """List available subjects in the knowledge base"""
    subjects = ['Biology', 'Chemistry', 'Physics', 'Mathematics', 'Computer_science']
    subject_info = {}
    
    for subject in subjects:
        if subject in vector_stores:
            # Get some basic info about each subject's vector store
            subject_info[subject] = {
                "available": True,
                "description": f"Knowledge base for {subject} with enhanced source tracking"
            }
        else:
            subject_info[subject] = {
                "available": False,
                "description": f"No data available for {subject}"
            }
    
    return {
        "subjects": subject_info,
        "total_subjects": len([s for s in subject_info.values() if s["available"]]),
        "source_tracking": "Each query result includes file names and section information"
    }

if __name__ == "__main__":
    print("\n🚀 Starting LangChain RAG API server...")
    print("📚 Features:")
    print("   - Enhanced source tracking with file names and sections")
    print("   - Multi-subject knowledge base")
    print("   - LangChain-based implementation")
    print("   - Structured response format with citations")
    print("\n🌐 Access the API at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

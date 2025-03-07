import threading
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Import your functions from llm.py (make sure llm.py is in the same directory or in your PYTHONPATH)
from llm import load_llm, load_indices_tools, load_tools, query_agent

# Initialize LLM, load indices and tools
llm = load_llm()
index_set = load_indices_tools()
tools = load_tools(llm, index_set)

# Create the FastAPI application
app = FastAPI()

# Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the LlamaIndex query agent API"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Process the query using your query_agent function
    response = query_agent(llm, tools, request.query)
    return {"response": response}

# Function to start the FastAPI server (using uvicorn)
def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Run the FastAPI server in a background thread so that Streamlit can also run in the same process
def run_api_server():
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    # Wait briefly to allow the server to start
    time.sleep(2)

# ---- Streamlit Frontend ----
import streamlit as st
import requests

def main():
    st.title("LLM Query Agent Web App")
    query_text = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        if query_text:
            try:
                # Send the query to the FastAPI endpoint
                res = requests.post("http://127.0.0.1:8000/query", json={"query": query_text})
                if res.status_code == 200:
                    st.write("Agent Response:")
                    st.write(res.json()["response"])
                else:
                    st.error(f"API Error: {res.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

if __name__ == "__main__":
    # Start the FastAPI server in the background
    run_api_server()
    # Run the Streamlit app
    main()

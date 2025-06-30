from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate as ReactPromptTemplate
import os
from pathlib import Path

subjects = ['Biology', 'Chemistry', 'Physics', 'Mathematics', 'Computer_science']

def initialise_llm():
    """Initialize the ChatOllama LLM"""
    llm = ChatOllama(model="tinyllama:latest", request_timeout=60)
    return llm

def load_vector_stores():
    """Load and create vector stores for each subject with enhanced metadata"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    vector_stores = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Match the chunk size from index.ipynb
        chunk_overlap=100,
        add_start_index=True,
    )
    
    for subject in subjects:
        # Create vector store for this subject
        vector_store = InMemoryVectorStore(embeddings)
        
        # Load documents for this subject
        subject_file_path = f"./data/test/{subject}.txt"
        
        if os.path.exists(subject_file_path):
            # Load the specific subject file
            with open(subject_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "subject": subject,
                    "document_name": f"{subject}.txt",
                    "source": subject,
                    "file_path": subject_file_path
                }
            )
            
            # Split the document into chunks
            splits = text_splitter.split_documents([doc])
            
            # Add section information to each chunk
            for i, split in enumerate(splits):
                split.metadata.update({
                    "chunk_id": i,
                    "section": f"Section {i+1}",
                    "total_chunks": len(splits)
                })
            
            # Add documents to vector store
            vector_store.add_documents(splits)
            print(f"Loaded {len(splits)} chunks for {subject}")
        
        vector_stores[subject] = vector_store
    
    return vector_stores

class RAGState(TypedDict):
    """State for RAG workflow"""
    question: str
    context: List[Document]
    answer: str
    sources: List[str]

def create_rag_tools(llm, vector_stores):
    """Create RAG tools for each subject that return source information"""
    tools = []
    
    # Template that emphasizes source information
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

IMPORTANT: You must include source information in your answer. For each piece of information you use, 
mention the document name and section it came from.

Context with Sources:
{context_with_sources}

Question: {question}

Answer (include sources):"""

    prompt = PromptTemplate.from_template(template)
    
    for subject in subjects:
        vector_store = vector_stores[subject]
        
        def create_retrieve_func(vs, subj):
            def retrieve(state: RAGState):
                retrieved_docs = vs.similarity_search(state["question"], k=3)
                sources = []
                for doc in retrieved_docs:
                    source_info = f"{doc.metadata.get('document_name', 'Unknown')} - {doc.metadata.get('section', 'Unknown section')}"
                    sources.append(source_info)
                return {"context": retrieved_docs, "sources": sources}
            return retrieve
        
        def create_generate_func(llm_instance, subj):
            def generate(state: RAGState):
                # Format context with source information
                context_with_sources = ""
                for i, doc in enumerate(state["context"]):
                    source_info = f"[Source: {doc.metadata.get('document_name', 'Unknown')} - {doc.metadata.get('section', 'Unknown section')}]"
                    context_with_sources += f"\n\nDocument {i+1} {source_info}:\n{doc.page_content}"
                
                messages = prompt.invoke({
                    "question": state["question"], 
                    "context_with_sources": context_with_sources
                })
                response = llm_instance.invoke(messages)
                
                # Format final answer with sources
                sources_list = state.get("sources", [])
                sources_text = " | ".join(sources_list) if sources_list else "No sources found"
                final_answer = f"{response.content}\n\n[Sources: {sources_text}]"
                
                return {
                    "answer": final_answer,
                    "context": state["context"],
                    "sources": sources_list
                }
            return generate
        
        # Create RAG graph for this subject
        graph_builder = StateGraph(RAGState)
        retrieve_func = create_retrieve_func(vector_store, subject)
        generate_func = create_generate_func(llm, subject)
        
        graph_builder.add_sequence([retrieve_func, generate_func])
        graph_builder.add_edge(START, retrieve_func.__name__)
        rag_graph = graph_builder.compile()
        
        # Create tool function
        def create_tool_func(graph, subj):
            @tool
            def query_tool(question: str) -> str:
                """Query the knowledge base for information about a specific subject.
                
                This tool searches through documents and returns answers with source citations
                including document names and sections.
                
                Args:
                    question: The question to ask about the subject
                    
                Returns:
                    Answer with source information including document names and sections
                """
                try:
                    result = graph.invoke({"question": question})
                    return result["answer"]
                except Exception as e:
                    return f"Error querying {subj} knowledge base: {str(e)}"
            
            # Update the tool's name and description after creation
            query_tool.name = f"query_{subj.lower()}"
            query_tool.description = f"Query the {subj} knowledge base for information about {subj.lower()}. Returns answers with source citations including document names and sections."
            
            return query_tool
        
        tool_func = create_tool_func(rag_graph, subject)
        tools.append(tool_func)
    
    return tools

def create_agent(llm, tools):
    """Create a ReAct agent with all the RAG tools"""
    
    # Create a custom prompt that emphasizes source citation
    react_prompt = ReactPromptTemplate.from_template("""
You are an intelligent agent that can answer questions about various subjects including Biology, Chemistry, Physics, Mathematics, and Computer Science.

You have access to specialized tools for each subject that will provide you with information along with source citations.

IMPORTANT INSTRUCTIONS:
1. When you use any tool, the tool will return information with source citations
2. You MUST include these source citations in your final answer
3. Always mention which document(s) and section(s) the information came from
4. If multiple sources are used, list all of them
5. Format your final answer clearly with the sources at the end

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [Your answer here] [Sources: list all sources used]

Question: {input}
Thought: {agent_scratchpad}
""")
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    return agent_executor

def query_agent(agent, query):
    """Query the agent and return the response"""
    try:
        result = agent.invoke({"input": query})
        return result
    except Exception as e:
        return {"output": f"Error processing query: {str(e)}"}

def dummy_query_agent(agent, query):
    """Dummy query for testing (similar to llm.py)"""
    try:
        result = query_agent(agent, query)
        print(f"Query: {query}")
        print(f"Response: {result.get('output', 'No output')}")
        return result
    except Exception as e:
        print(f"Error in dummy query: {str(e)}")
        return None

# Main functions to match llm.py interface
def load_llm():
    """Alias for initialise_llm to match llm.py interface"""
    return initialise_llm()

def load_indices_tools():
    """Alias for load_vector_stores to match llm.py interface"""
    return load_vector_stores()

def load_tools(llm, vector_stores):
    """Alias for create_rag_tools to match llm.py interface"""
    return create_rag_tools(llm, vector_stores)

def load_agent(llm, tools):
    """Alias for create_agent to match llm.py interface"""
    return create_agent(llm, tools)

# Test function
if __name__ == "__main__":
    print("Testing LangChain RAG implementation...")
    
    # Initialize components
    llm = load_llm()
    print("✓ Loaded LLM")
    
    vector_stores = load_indices_tools()
    print("✓ Loaded vector stores")
    
    tools = load_tools(llm, vector_stores)
    print("✓ Created RAG tools")
    
    agent = load_agent(llm, tools)
    print("✓ Created agent")
    
    # Test queries
    test_queries = [
        "What is computer science?",
        "Explain basic concepts in biology",
        "What are the fundamental principles of physics?"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        result = dummy_query_agent(agent, query)

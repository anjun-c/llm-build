subjects = ['Biology', 'Chemistry', 'Physics', 'Mathematics', 'Computer_science']

def load_indices_tools():
    # Load indices from disk
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index_set = {}

    for subject in subjects:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{subject}"
        )
        cur_index = load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )
        index_set[subject] = cur_index

    return index_set

def load_tools(llm, index_set):
    # Load tools
    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[subject].as_query_engine(llm=llm),
            metadata=ToolMetadata(
                name=f"vector_index_{subject}",
                description=f"useful for when you want to answer queries about the {subject} subject",
            ),
        )
        for subject in subjects
    ]

    from llama_index.core.query_engine import SubQuestionQueryEngine

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=llm,
    )

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine",
            description="useful for when you want to answer queries that require analyzing multiple subjects",
        ),
    )

    tools = individual_query_engine_tools + [query_engine_tool]

    return tools

def load_llm():
    from llama_index.llms.ollama import Ollama

    llm = Ollama(model="deepseek-r1:8b", request_timeout=60)

    return llm

def load_agent(llm, tools):
    custom_prompt = """
    You are an intelligent agent that uses chain-of-thought reasoning internally but must output only one final answer.
    
    When you are ready, output exactly one line in the following format and nothing else:
    Answer: <Your final answer here.>
    """

    from llama_index.core.agent import ReActAgent
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=False, custom_prompt=custom_prompt)

    return agent


def dummy_query_agent(llm, tools, agent, input):
    agent.chat(input)

def query_agent(llm, tools, agent, input):
    import re
    
    response = agent.chat(input)
    print(f"Response is: {response}")
    
    # Convert response to string if needed
    response_str = str(response)
    
    # Remove the chain-of-thought block enclosed in <think> ... </think>
    cleaned = re.sub(r"<think>.*?</think>\s*", "", response_str, flags=re.DOTALL)
    
    # Extract the final answer line if it starts with "Answer:"
    match = re.search(r"Answer:\s*(.*)", cleaned, flags=re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
    else:
        final_answer = cleaned.strip()
    
    print(f"Final answer is: {final_answer}")

    return final_answer

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

    # from llama_index.core.query_engine import SubQuestionQueryEngine

    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=individual_query_engine_tools,
    #     llm=llm,
    # )

    # query_engine_tool = QueryEngineTool(
    #     query_engine=query_engine,
    #     metadata=ToolMetadata(
    #         name="sub_question_query_engine",
    #         description="useful for when you want to answer queries that require analyzing multiple subjects",
    #     ),
    # )

    tools = individual_query_engine_tools #+ [query_engine_tool]

    return tools

def load_llm():
    from llama_index.llms.ollama import Ollama

    llm = Ollama(model="tinyllama:latest", request_timeout=60)

    return llm

def load_agent(llm, tools):
    custom_prompt = """
    You are an intelligent agent that can answer questions about various subjects. 
    You have access to a large amount of information about Biology, Chemistry, Physics, Mathematics, and Computer Science, 
    and you also have vector index tools available for certain domains. 
    However, if all tools do not return any relevant information, or if a tool for a specific domain does not exist, 
    you should answer the question based solely on your general knowledge without using any vector indices. 
    If your tool queries do not yield relevant results after a few attempts, then provide an answer based solely on your general knowledge.
    You can ask clarifying questions if you need more information to answer a question, and you may ask up to 20 clarifying questions.

    When you are ready, output exactly in the following format and nothing else:
    Answer: <Your final answer here.>
    """


    from llama_index.core.agent import ReActAgent
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, custom_prompt=custom_prompt, maximum_interactions=20)

    return agent

def dummy_query_agent(llm, tools, agent, input):
    agent.chat(input)
    # llm.complete(input)

def query_agent(llm, tools, agent, input):
    
    response = agent.chat(input)

    return response
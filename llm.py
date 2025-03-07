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

    llm = Ollama(model="deepseek-r1:8b")

    return llm

def query_agent(llm, tools, input):
    custom_prompt = """
    You are an intelligent agent that answers queries by following a strict chain-of-thought format.
    For every step, please output your reasoning and your tool calls using exactly the following structure:

    Thought: <Your internal reasoning for this step.>
    Action: <The tool name you want to use (or output "None" if no tool is used).>
    Action Input: <The JSON input to pass to the tool (or "{}" if no tool is used).>

    When you are ready to provide your final answer, output only:
    Answer: <Your final answer here.>

    Do not output any extra text or commentary outside of this structure.
    """

    from llama_index.core.agent import ReActAgent

    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, prompt_template=custom_prompt)

    response = agent.chat(input)
    
    return response
def initialise_llm():
    from langchain_community.chat_models import OllamaChat
    
    #llm = ChatOllama(model="llama3.1:latest", request_timeout=60)
    llm = OllamaChat(model="tinyllama:latest", request_timeout=60)

    return llm

def ingest_data():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import DirectoryLoader

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)

    loader = DirectoryLoader(
        path="./data/test",
        glob="**/*.txt",
        show_progress=True,
    )

    documents = loader.load()
    print(f"Total characters: {len(docs[0].page_content)}")

    print(docs[0].page_content[:500])
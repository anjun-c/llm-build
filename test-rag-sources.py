"""
Test script to demonstrate the enhanced RAG source tracking functionality
without requiring Ollama to be running.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def test_source_tracking():
    """Test the source tracking functionality"""
    print("Testing RAG Source Tracking Functionality...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        add_start_index=True,
    )
    
    # Test with Computer Science file
    subject = "Computer_science"
    subject_file_path = f"./data/test/{subject}.txt"
    
    if os.path.exists(subject_file_path):
        print(f"\n✓ Found {subject}.txt file")
        
        # Load the file
        with open(subject_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✓ Loaded content: {len(content)} characters")
        
        # Create document with enhanced metadata
        doc = Document(
            page_content=content,
            metadata={
                "subject": subject,
                "document_name": f"{subject}.txt",
                "source": subject,
                "file_path": subject_file_path
            }
        )
        
        # Split into chunks
        splits = text_splitter.split_documents([doc])
        
        # Add section information to each chunk
        for i, split in enumerate(splits):
            split.metadata.update({
                "chunk_id": i,
                "section": f"Section {i+1}",
                "total_chunks": len(splits)
            })
        
        print(f"✓ Split into {len(splits)} chunks with enhanced metadata")
        
        # Create vector store and add documents
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(splits)
        
        print(f"✓ Added documents to vector store")
        
        # Test retrieval with source information
        test_query = "What is computer science?"
        retrieved_docs = vector_store.similarity_search(test_query, k=3)
        
        print(f"\n--- Retrieval Results for: '{test_query}' ---")
        print(f"Retrieved {len(retrieved_docs)} documents with source information:")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.metadata.get('document_name', 'Unknown')}")
            print(f"  Section: {doc.metadata.get('section', 'Unknown section')}")
            print(f"  Subject: {doc.metadata.get('subject', 'Unknown')}")
            print(f"  Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
            print(f"  Content Preview: {doc.page_content[:200]}...")
            
            # Format source information as it would appear in responses
            source_info = f"{doc.metadata.get('document_name', 'Unknown')} - {doc.metadata.get('section', 'Unknown section')}"
            print(f"  Formatted Source: [{source_info}]")
        
        # Demonstrate how sources would be included in final response
        sources = []
        for doc in retrieved_docs:
            source_info = f"{doc.metadata.get('document_name', 'Unknown')} - {doc.metadata.get('section', 'Unknown section')}"
            sources.append(source_info)
        
        sources_text = " | ".join(sources)
        print(f"\n--- Final Response Format ---")
        print(f"Answer: [This would be the LLM's answer based on the retrieved content]")
        print(f"[Sources: {sources_text}]")
        
        print(f"\n✅ Source tracking test completed successfully!")
        print(f"✅ Each retrieved chunk includes:")
        print(f"   - Document name (file name)")
        print(f"   - Section information (chunk position)")
        print(f"   - Subject classification")
        print(f"   - Metadata for traceability")
        
    else:
        print(f"❌ File not found: {subject_file_path}")

if __name__ == "__main__":
    test_source_tracking()

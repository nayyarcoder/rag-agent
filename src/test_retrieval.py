import chromadb
import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_collection():
    # Initialize ChromaDB client
    db_path = Path("db")
    client = chromadb.PersistentClient(path=str(db_path))
    
    # List all collections
    collections = client.list_collections()
    logger.info(f"Found collections: {[c.name for c in collections]}")
    
    for collection in collections:
        count = collection.count()
        logger.info(f"Collection '{collection.name}' has {count} documents")
        
        # Get collection metadata
        metadata = collection.get()
        logger.info(f"Collection metadata: {metadata}")
        
        # Try to get some documents
        if count > 0:
            results = collection.get(limit=1)
            logger.info(f"Sample document from collection: {results}")

def test_retriever():
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings,
        collection_name="documents"
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Test queries
    test_queries = [
        "test query",
        "what is this document about",
        "tell me about the content",
        "what is in the sample text"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        try:
            docs = retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info(f"Metadata: {doc.metadata}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")

if __name__ == "__main__":
    logger.info("Testing ChromaDB collection...")
    test_collection()
    
    logger.info("\nTesting retriever functionality...")
    test_retriever() 
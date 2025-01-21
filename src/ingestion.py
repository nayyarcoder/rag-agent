from typing import List, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import chromadb
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "db"
    ):
        """Initialize the document ingester."""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from a file path."""
        logger.info(f"Loading document from {file_path}")
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents = [Document(page_content=text, metadata={"source": file_path})]
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return []

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting them into chunks."""
        logger.info("Processing documents")
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []

    def ingest_files(self, file_paths: List[str], collection_name: str = "documents") -> Optional[Chroma]:
        """Ingest files from the provided file paths into a vector store."""
        logger.info("Starting document ingestion process")
        
        # Initialize vector store
        logger.info("Initializing vector store")
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Process each file
        logger.info(f"Processing {len(file_paths)} files")
        for file_path in file_paths:
            if os.path.isfile(file_path):
                logger.info(f"Processing file: {file_path}")
                documents = self.load_document(file_path)
                if documents:
                    chunks = self.process_documents(documents)
                    if chunks:
                        logger.info(f"Adding {len(chunks)} chunks to vector store")
                        try:
                            vectorstore.add_documents(chunks)
                            logger.info(f"Successfully added chunks from {file_path} to vector store")
                        except Exception as e:
                            logger.error(f"Error adding chunks to vector store: {str(e)}")
        
        logger.info("Document ingestion process completed")
        return vectorstore

def main():
    """Main function to run the ingestion process."""
    ingester = DocumentIngester()
    ingester.ingest_files()

if __name__ == "__main__":
    main() 
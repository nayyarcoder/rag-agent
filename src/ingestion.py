from typing import List, Optional
from pathlib import Path
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import chromadb
import logging

from config import get_config
from document_loaders import DocumentLoaderFactory
from exceptions import (
    DocumentLoadError, 
    DocumentProcessingError,
    VectorStoreError,
    EmbeddingError,
    UnsupportedDocumentTypeError
)

logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        max_workers: int = 4
    ):
        """Initialize the document ingester with enhanced configuration."""
        self.config = get_config()
        
        # Override config with provided parameters
        self.chunk_size = chunk_size or self.config.document.chunk_size
        self.chunk_overlap = chunk_overlap or self.config.document.chunk_overlap
        self.embedding_model = embedding_model or self.config.embedding.model_name
        self.persist_directory = persist_directory or self.config.vector_store.persist_directory
        self.max_workers = max_workers
        
        # Setup logging if not already configured
        if not logger.handlers:
            self.config.setup_logging()
        
        logger.info(f"Initializing DocumentIngester with:")
        logger.info(f"  - Chunk size: {self.chunk_size}")
        logger.info(f"  - Chunk overlap: {self.chunk_overlap}")
        logger.info(f"  - Embedding model: {self.embedding_model}")
        logger.info(f"  - Persist directory: {self.persist_directory}")
        logger.info(f"  - Max workers: {self.max_workers}")
        
        # Initialize embeddings with retry logic
        self._initialize_embeddings()
        
        # Initialize ChromaDB client
        self._initialize_chroma_client()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def _initialize_embeddings(self) -> None:
        """Initialize embeddings with retry logic."""
        for attempt in range(self.config.embedding.max_retries):
            try:
                logger.info(f"Initializing embeddings (attempt {attempt + 1})")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model
                )
                logger.info("Embeddings initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings (attempt {attempt + 1}): {str(e)}")
                if attempt == self.config.embedding.max_retries - 1:
                    raise EmbeddingError(f"Failed to initialize embeddings after {self.config.embedding.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _initialize_chroma_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized with path: {self.persist_directory}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {str(e)}")

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from a file path with enhanced error handling."""
        logger.info(f"Loading document from {file_path}")
        
        try:
            # Create appropriate loader using factory
            loader = DocumentLoaderFactory.create_loader(file_path)
            documents = loader.load()
            
            if not documents:
                raise DocumentLoadError(file_path, "No documents were extracted")
            
            # Add processing metadata
            for doc in documents:
                doc.metadata.update({
                    "loaded_at": time.time(),
                    "loader_type": type(loader).__name__,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except (DocumentLoadError, UnsupportedDocumentTypeError) as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading document {file_path}: {str(e)}")
            return []

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting them into chunks with enhanced error handling."""
        if not documents:
            logger.warning("No documents provided for processing")
            return []
        
        logger.info(f"Processing {len(documents)} documents")
        
        try:
            # Filter out documents with empty content
            valid_documents = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_documents:
                raise DocumentProcessingError("All documents have empty content")
            
            if len(valid_documents) < len(documents):
                logger.warning(f"Filtered out {len(documents) - len(valid_documents)} empty documents")
            
            # Split documents into chunks
            start_time = time.time()
            split_docs = self.text_splitter.split_documents(valid_documents)
            processing_time = time.time() - start_time
            
            # Add chunk metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                    "processed_at": time.time(),
                    "processing_time": processing_time
                })
            
            logger.info(f"Split {len(valid_documents)} documents into {len(split_docs)} chunks in {processing_time:.2f}s")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    def _process_single_file(self, file_path: str) -> tuple[str, List[Document], Optional[str]]:
        """Process a single file and return results."""
        try:
            if not os.path.isfile(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return file_path, [], error_msg
            
            documents = self.load_document(file_path)
            if documents:
                chunks = self.process_documents(documents)
                return file_path, chunks, None
            else:
                return file_path, [], "No documents loaded"
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return file_path, [], error_msg

    def ingest_files(self, file_paths: List[str], collection_name: str = "documents") -> Optional[Chroma]:
        """Ingest files with concurrent processing and enhanced error handling."""
        if not file_paths:
            logger.warning("No file paths provided for ingestion")
            return None
        
        logger.info(f"Starting document ingestion for {len(file_paths)} files")
        start_time = time.time()
        
        try:
            # Initialize vector store
            vectorstore = self._initialize_vectorstore(collection_name)
            
            # Process files concurrently
            all_chunks = []
            failed_files = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all file processing tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path 
                    for file_path in file_paths
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path, chunks, error = future.result()
                    
                    if error:
                        failed_files.append((file_path, error))
                    else:
                        all_chunks.extend(chunks)
                        logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            
            # Log processing summary
            total_time = time.time() - start_time
            logger.info(f"File processing completed in {total_time:.2f}s:")
            logger.info(f"  - Successfully processed: {len(file_paths) - len(failed_files)} files")
            logger.info(f"  - Failed: {len(failed_files)} files")
            logger.info(f"  - Total chunks: {len(all_chunks)}")
            
            if failed_files:
                logger.warning("Failed to process the following files:")
                for file_path, error in failed_files:
                    logger.warning(f"  - {file_path}: {error}")
            
            # Add chunks to vector store if any were processed
            if all_chunks:
                logger.info(f"Adding {len(all_chunks)} chunks to vector store")
                self._add_chunks_to_vectorstore(vectorstore, all_chunks)
                logger.info("Document ingestion completed successfully")
            else:
                logger.warning("No chunks to add to vector store")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            raise VectorStoreError(f"Document ingestion failed: {str(e)}")
    
    def _initialize_vectorstore(self, collection_name: str) -> Chroma:
        """Initialize vector store."""
        try:
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector store initialized with collection: {collection_name}")
            return vectorstore
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
    
    def _add_chunks_to_vectorstore(self, vectorstore: Chroma, chunks: List[Document]) -> None:
        """Add chunks to vector store with batch processing."""
        batch_size = self.config.embedding.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Filter complex metadata that might cause issues
                filtered_batch = filter_complex_metadata(batch)
                vectorstore.add_documents(filtered_batch)
                logger.debug(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
                # Try to add documents individually to identify problematic ones
                for j, doc in enumerate(batch):
                    try:
                        filtered_doc = filter_complex_metadata([doc])
                        vectorstore.add_documents(filtered_doc)
                    except Exception as doc_error:
                        logger.error(f"Failed to add document {i + j}: {str(doc_error)}")
    
    def get_ingestion_stats(self) -> dict:
        """Get statistics about the ingestion process."""
        try:
            collections = self.client.list_collections()
            stats = {
                "total_collections": len(collections),
                "collections": []
            }
            
            for collection in collections:
                collection_stats = {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.get(limit=1)
                }
                stats["collections"].append(collection_stats)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> dict:
        """Perform health check on the ingestion system."""
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        # Check embeddings
        try:
            test_text = "health check"
            embeddings = self.embeddings.embed_query(test_text)
            health["checks"]["embeddings"] = {"status": "ok", "dimension": len(embeddings)}
        except Exception as e:
            health["checks"]["embeddings"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check ChromaDB
        try:
            collections = self.client.list_collections()
            health["checks"]["chromadb"] = {"status": "ok", "collections": len(collections)}
        except Exception as e:
            health["checks"]["chromadb"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.persist_directory)
            health["checks"]["disk_space"] = {
                "status": "ok",
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2)
            }
        except Exception as e:
            health["checks"]["disk_space"] = {"status": "error", "error": str(e)}
        
        return health

def main():
    """Main function to run the ingestion process."""
    try:
        # Initialize configuration and logging
        config = get_config()
        config.validate()
        config.setup_logging()
        
        logger.info("Starting RAG Agent document ingestion")
        logger.info(f"Supported file types: {DocumentLoaderFactory.get_supported_extensions()}")
        
        # Initialize ingester
        ingester = DocumentIngester()
        
        # Perform health check
        health = ingester.health_check()
        logger.info(f"Health check: {health}")
        
        if health["status"] != "healthy":
            logger.error("System health check failed - aborting ingestion")
            return
        
        # Example: ingest files from data directory if it exists
        data_dir = Path("data")
        if data_dir.exists():
            file_paths = []
            for ext in DocumentLoaderFactory.get_supported_extensions():
                file_paths.extend(data_dir.glob(f"*{ext}"))
            
            if file_paths:
                file_paths = [str(p) for p in file_paths]
                logger.info(f"Found {len(file_paths)} files to ingest")
                vectorstore = ingester.ingest_files(file_paths)
                
                # Print stats
                stats = ingester.get_ingestion_stats()
                logger.info(f"Ingestion stats: {stats}")
            else:
                logger.info("No supported files found in data directory")
        else:
            logger.info("No data directory found - skipping file ingestion")
            
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
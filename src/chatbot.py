import os
from dotenv import load_dotenv
from typing import List, Optional
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging

from config import get_config, get_available_llm_providers
from exceptions import VectorStoreError, EmbeddingError, ConfigurationError

# Try to import streamlit, but don't fail if it's not available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        """Initialize RAG chatbot with enhanced configuration and error handling."""
        self.config = get_config()
        
        # Override LLM provider if specified
        if llm_provider:
            self.config.llm.provider = llm_provider
        
        # Setup logging if not already configured
        if not logger.handlers:
            self.config.setup_logging()
        
        # Use provided parameters or fall back to config
        self.persist_directory = persist_directory or self.config.vector_store.persist_directory
        self.collection_name = collection_name or self.config.vector_store.collection_name
        self.model_name = model_name or self.config.llm.model_name
        self.embedding_model = embedding_model or self.config.embedding.model_name
        
        logger.info(f"Initializing RAGChatbot:")
        logger.info(f"  - Collection: {self.collection_name}")
        logger.info(f"  - Persist directory: {self.persist_directory}")
        logger.info(f"  - LLM model: {self.model_name}")
        logger.info(f"  - Embedding model: {self.embedding_model}")
        
        # Initialize components with enhanced error handling
        self._initialize_embeddings()
        self._initialize_chromadb()
        self._initialize_vectorstore()
        self._initialize_llm()
        self._setup_rag_chain()
        
        logger.info("RAGChatbot initialization completed successfully")
    
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
    
    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client with validation."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized with path: {self.persist_directory}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {str(e)}")
    
    def _initialize_vectorstore(self) -> None:
        """Initialize vector store and validate collection."""
        try:
            # Check if collection exists and has documents
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            logger.info(f"Collection '{self.collection_name}' found with {count} documents")
            
            if count == 0:
                raise VectorStoreError(f"Collection '{self.collection_name}' exists but is empty")
            
            # Initialize vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test retrieval functionality
            self._test_retrieval()
            
        except chromadb.errors.InvalidCollectionException:
            raise VectorStoreError(f"Collection '{self.collection_name}' does not exist")
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            raise VectorStoreError(f"Error initializing vector store: {str(e)}")
    
    def _test_retrieval(self) -> None:
        """Test retrieval functionality."""
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
            logger.info("Testing retriever functionality...")
            test_results = retriever.get_relevant_documents("test query")
            logger.info(f"Retriever test returned {len(test_results)} documents")
        except Exception as e:
            logger.error(f"Error testing retriever: {str(e)}")
            raise VectorStoreError(f"Retrieval test failed: {str(e)}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM with configuration supporting multiple providers."""
        try:
            provider = self.config.llm.provider.lower()
            providers_info = get_available_llm_providers()
            
            if provider not in providers_info:
                raise ConfigurationError(f"Unsupported LLM provider: {provider}. Supported providers: {list(providers_info.keys())}")
            
            provider_info = providers_info[provider]
            
            # Validate API key for providers that require it
            if provider_info.get("requires_api_key", True):
                api_key = self.config.llm.api_key
                if not api_key:
                    env_var = provider_info.get("env_var", f"{provider.upper()}_API_KEY")
                    raise ConfigurationError(f"{env_var} environment variable not set for provider '{provider}'")
            
            # Prepare model identifier for LiteLLM
            if provider == "ollama":
                # For Ollama, use the model name directly since it's served locally
                model_identifier = self.model_name
                api_base = self.config.llm.api_base or provider_info.get("default_api_base", "http://localhost:11434")
            else:
                # For cloud providers, prefix with provider name
                model_identifier = f"{provider}/{self.model_name}"
                api_base = self.config.llm.api_base
            
            # Initialize LiteLLM with provider-specific configuration
            llm_kwargs = {
                "model": model_identifier,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "timeout": self.config.llm.timeout_seconds,
            }
            
            # Set API key if required
            if provider_info.get("requires_api_key", True) and self.config.llm.api_key:
                llm_kwargs["api_key"] = self.config.llm.api_key
            
            # Set API base if provided
            if api_base:
                llm_kwargs["api_base"] = api_base
            
            self.llm = ChatLiteLLM(**llm_kwargs)
            
            logger.info(f"LLM initialized with provider: {provider}, model: {self.model_name}")
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to initialize LLM: {str(e)}")
    
    def _setup_rag_chain(self) -> None:
        """Setup RAG chain with enhanced prompt."""
        try:
            # Create enhanced prompt
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on the provided document context.
                
                Your responses should be:
                1. Based ONLY on the information found in the provided context
                2. Clear, concise, and well-structured
                3. Include relevant details from the source documents
                4. If the answer cannot be found in the context, say "I cannot find information about that in the provided documents."
                5. If you need more context to provide a complete answer, say so.
                6. When possible, mention which type of document the information came from (e.g., "According to the CSV data..." or "The HTML documentation states...")
                
                Context from documents: {context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            # Create the RAG chain with enhanced retriever
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.config.vector_store.max_results,
                        "score_threshold": self.config.vector_store.similarity_threshold
                    }
                ),
                combine_docs_chain_kwargs={
                    "prompt": self.prompt
                },
                return_source_documents=True,
                memory=None,
                get_chat_history=lambda h: h
            )
            logger.info("RAG chain setup completed")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to setup RAG chain: {str(e)}")

    @staticmethod
    def get_available_collections(persist_directory: Optional[str] = None) -> List[dict]:
        """Get a list of available collections with enhanced information."""
        config = get_config()
        persist_dir = persist_directory or config.vector_store.persist_directory
        
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collections = client.list_collections()
            collection_info = []
            
            for collection in collections:
                try:
                    count = collection.count()
                    # Get sample metadata to understand document types
                    sample_data = collection.get(limit=3)
                    
                    # Extract file types from metadata
                    file_types = set()
                    if sample_data and sample_data.get('metadatas'):
                        for metadata in sample_data['metadatas']:
                            if metadata and 'file_type' in metadata:
                                file_types.add(metadata['file_type'])
                            elif metadata and 'source' in metadata:
                                # Extract from file extension
                                source = metadata['source']
                                if '.' in source:
                                    ext = source.split('.')[-1].lower()
                                    file_types.add(ext)
                    
                    collection_info.append({
                        'name': collection.name,
                        'count': count,
                        'file_types': list(file_types),
                        'sample_sources': [m.get('source', 'Unknown') for m in (sample_data.get('metadatas') or []) if m][:3]
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for collection {collection.name}: {str(e)}")
                    collection_info.append({
                        'name': collection.name,
                        'count': 0,
                        'file_types': [],
                        'sample_sources': [],
                        'error': str(e)
                    })
            
            logger.info(f"Found {len(collection_info)} collections")
            return [c for c in collection_info if c['count'] > 0]
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get detailed statistics for current collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            # Get sample documents to analyze
            sample_data = collection.get(limit=10)
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "file_types": {},
                "sources": [],
                "chunk_info": {
                    "average_chunk_size": 0,
                    "total_chunks": count
                }
            }
            
            if sample_data and sample_data.get('metadatas'):
                # Analyze file types
                file_type_counts = {}
                chunk_sizes = []
                sources = set()
                
                for metadata in sample_data['metadatas']:
                    if metadata:
                        # File type analysis
                        file_type = metadata.get('file_type', 'unknown')
                        file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                        
                        # Source tracking
                        if 'source' in metadata:
                            sources.add(metadata['source'])
                        
                        # Chunk size analysis
                        if 'chunk_size' in metadata:
                            chunk_sizes.append(metadata['chunk_size'])
                
                stats['file_types'] = file_type_counts
                stats['sources'] = list(sources)[:10]  # Limit to 10 sources
                
                if chunk_sizes:
                    stats['chunk_info']['average_chunk_size'] = sum(chunk_sizes) // len(chunk_sizes)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
        
    def get_response(self, question: str, chat_history: List[tuple]) -> str:
        """Get a response from the chatbot with enhanced error handling and logging."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question}")
            logger.info(f"Current collection: {self.collection_name}")
            
            # Validate input
            if not question or not question.strip():
                return "Please provide a valid question."
            
            # Convert chat history to Message objects
            messages = []
            for human_msg, ai_msg in chat_history[-5:]:  # Limit to last 5 exchanges
                messages.append(HumanMessage(content=str(human_msg)))
                messages.append(AIMessage(content=str(ai_msg)))
            
            # Get response from the chain with timeout handling
            logger.info("Retrieving documents from vector store...")
            retrieval_start = time.time()
            
            response = self.chain.invoke({
                "question": question,
                "chat_history": messages
            })
            
            retrieval_time = time.time() - retrieval_start
            total_time = time.time() - start_time
            
            # Log performance metrics
            source_docs = response.get("source_documents", [])
            logger.info(f"Retrieved {len(source_docs)} chunks in {retrieval_time:.2f}s")
            logger.info(f"Total response time: {total_time:.2f}s")
            
            # Enhanced document display in Streamlit
            if HAS_STREAMLIT and hasattr(st, 'expander'):  # Check if running in Streamlit context
                with st.expander("📄 View Retrieved Document Chunks", expanded=False):
                    if not source_docs:
                        st.warning("No relevant chunks were retrieved for this query.")
                        
                        # Display collection stats for debugging
                        stats = self.get_collection_stats()
                        st.info(f"Collection '{self.collection_name}' has {stats.get('total_documents', 0)} total documents")
                        
                        if stats.get('file_types'):
                            st.write("**Available document types:**")
                            for file_type, count in stats['file_types'].items():
                                st.write(f"- {file_type}: {count} documents")
                    else:
                        st.markdown("### Retrieved Document Chunks")
                        
                        for i, doc in enumerate(source_docs, 1):
                            with st.container():
                                # Enhanced metadata display
                                metadata = doc.metadata
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**Chunk {i}:**")
                                with col2:
                                    if 'file_type' in metadata:
                                        st.badge(metadata['file_type'].upper(), type="secondary")
                                
                                # Content preview (truncated if too long)
                                content = doc.page_content
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                
                                st.text_area(
                                    f"Content {i}",
                                    content,
                                    height=100,
                                    disabled=True,
                                    key=f"chunk_{i}_{hash(content)}"
                                )
                                
                                # Metadata display
                                if metadata:
                                    with st.expander(f"Metadata for Chunk {i}", expanded=False):
                                        for key, value in metadata.items():
                                            if key in ['source', 'file_type', 'chunk_index', 'loaded_at']:
                                                if key == 'loaded_at' and isinstance(value, (int, float)):
                                                    import datetime
                                                    value = datetime.datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                                                st.write(f"**{key}:** {value}")
                                
                                st.markdown("---")
            
            # Check if we got any source documents
            if not source_docs:
                # Provide helpful suggestions
                suggestions = []
                stats = self.get_collection_stats()
                
                if stats.get('file_types'):
                    file_types = list(stats['file_types'].keys())
                    suggestions.append(f"Available document types: {', '.join(file_types)}")
                
                if stats.get('sources'):
                    suggestions.append(f"Try asking about: {', '.join(stats['sources'][:3])}")
                
                suggestion_text = "\n\n".join(suggestions) if suggestions else ""
                
                return f"I cannot find any relevant information in the documents to answer your question. Please try rephrasing your question or ask about a different topic.\n\n{suggestion_text}"
            
            answer = response["answer"]
            
            # Log successful response
            logger.info(f"Successfully generated response ({len(answer)} characters)")
            
            return answer
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error in get_response after {error_time:.2f}s: {str(e)}")
            
            # Provide user-friendly error messages
            if "timeout" in str(e).lower():
                return "I'm sorry, but the request timed out. Please try again with a simpler question."
            elif "api" in str(e).lower():
                return "I'm experiencing difficulties with the language model. Please try again later."
            elif "embedding" in str(e).lower():
                return "I'm having trouble processing your question. Please try rephrasing it."
            else:
                return f"I encountered an error while processing your question. Please try again."
    
    def health_check(self) -> dict:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.time()
        }
        
        # Check embeddings
        try:
            test_text = "health check test"
            embeddings = self.embeddings.embed_query(test_text)
            health["checks"]["embeddings"] = {
                "status": "ok", 
                "dimension": len(embeddings),
                "model": self.embedding_model
            }
        except Exception as e:
            health["checks"]["embeddings"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check ChromaDB
        try:
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            health["checks"]["chromadb"] = {
                "status": "ok", 
                "collections": len(collections),
                "collection_names": collection_names
            }
        except Exception as e:
            health["checks"]["chromadb"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check current collection
        try:
            stats = self.get_collection_stats()
            health["checks"]["current_collection"] = {
                "status": "ok",
                "name": self.collection_name,
                "document_count": stats.get("total_documents", 0),
                "file_types": stats.get("file_types", {})
            }
        except Exception as e:
            health["checks"]["current_collection"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check LLM (basic)
        try:
            # Check provider configuration
            provider = self.config.llm.provider.lower()
            providers_info = get_available_llm_providers()
            provider_info = providers_info.get(provider, {})
            
            # Check API key for providers that require it
            api_key_configured = True
            if provider_info.get("requires_api_key", True):
                api_key = self.config.llm.api_key
                api_key_configured = bool(api_key)
            
            health["checks"]["llm"] = {
                "status": "ok" if api_key_configured else "warning",
                "provider": provider,
                "model": self.model_name,
                "api_key_configured": api_key_configured,
                "api_base": self.config.llm.api_base
            }
            
            if not api_key_configured and provider_info.get("requires_api_key", True):
                health["status"] = "degraded"
                
        except Exception as e:
            health["checks"]["llm"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        return health

def initialize_session_state():
    """Initialize session state variables."""
    if HAS_STREAMLIT:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = None

def main():
    if not HAS_STREAMLIT:
        print("Streamlit not available. This function requires Streamlit to run.")
        return
        
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 RAG Chatbot")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Get available collections
        available_collections = RAGChatbot.get_available_collections()
        if not available_collections:
            st.warning("No document collections found. Please ingest some documents first.")
            return
            
        collection_name = st.selectbox(
            "Select Document Collection",
            options=available_collections,
            help="Choose which document collection to query"
        )
        
        model_name = st.selectbox(
            "Model",
            options=["llama-3.1-8b-instant", "llama3-8b-8192", "gpt-4o-mini", "claude-3-haiku-20240307"],
            help="LLM model to use (provider will be auto-detected)"
        )
        
        if st.button("Initialize Chatbot"):
            with st.spinner("Initializing chatbot..."):
                try:
                    st.session_state.chatbot = RAGChatbot(
                        collection_name=collection_name,
                        model_name=model_name
                    )
                    st.success("Chatbot initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing chatbot: {str(e)}")
    
    # Chat interface
    if st.session_state.chatbot is None:
        st.info("Please initialize the chatbot using the sidebar.")
        return
    
    # Display chat history
    for human, ai in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("assistant"):
            st.write(ai)
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents"):
        with st.chat_message("human"):
            st.write(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(
                    question,
                    st.session_state.chat_history
                )
                st.write(response)
                
        # Update chat history
        st.session_state.chat_history.append((question, response))

if __name__ == "__main__":
    main() 
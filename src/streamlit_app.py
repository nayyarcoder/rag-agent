import streamlit as st
import tempfile
import os
import time
from pathlib import Path
from typing import Optional

from config import get_config
from ingestion import DocumentIngester
from chatbot import RAGChatbot
from document_loaders import DocumentLoaderFactory
from exceptions import (
    DocumentLoadError, 
    UnsupportedDocumentTypeError,
    VectorStoreError,
    ConfigurationError
)

# Initialize configuration
config = get_config()
config.setup_logging()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add DB_PATH constant
DB_PATH = Path(config.vector_store.persist_directory)
if not DB_PATH.exists():
    DB_PATH.mkdir(parents=True)

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temporary directory and return the path."""
    try:
        # Validate file type
        file_ext = Path(uploaded_file.name).suffix.lower()
        supported_extensions = DocumentLoaderFactory.get_supported_extensions()
        
        if file_ext not in supported_extensions:
            st.error(f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}")
            return None
        
        # Check file size
        file_size = len(uploaded_file.getvalue())
        max_size = config.document.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            st.error(f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed size ({config.document.max_file_size_mb} MB)")
            return None
        
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.success(f"Saved {uploaded_file.name} ({file_size / 1024:.1f} KB)")
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

def render_ingestion_tab():
    st.header("📚 Document Ingestion")
    st.write("Upload your documents and configure the ingestion parameters.")
    
    # Display supported file types prominently
    with st.expander("📋 Supported File Types", expanded=True):
        supported_extensions = DocumentLoaderFactory.get_supported_extensions()
        cols = st.columns(3)
        
        file_type_info = {
            '.pdf': ('📄', 'PDF Documents', 'Research papers, reports, books'),
            '.txt': ('📝', 'Text Files', 'Plain text documents'),
            '.docx': ('📘', 'Word Documents', 'Microsoft Word files'),
            '.doc': ('📘', 'Word Documents', 'Legacy Word files'),
            '.html': ('🌐', 'HTML Files', 'Web pages, documentation'),
            '.htm': ('🌐', 'HTML Files', 'Web pages, documentation'),
            '.csv': ('📊', 'CSV Files', 'Spreadsheet data, tables'),
            '.json': ('🔧', 'JSON Files', 'API responses, configuration'),
            '.md': ('📋', 'Markdown', 'Documentation, README files')
        }
        
        for i, ext in enumerate(supported_extensions):
            col = cols[i % 3]
            if ext in file_type_info:
                icon, name, desc = file_type_info[ext]
                col.write(f"{icon} **{name}** (`{ext}`)")
                col.caption(desc)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Ingestion Configuration")
        
        chunk_size = st.slider(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=config.document.chunk_size,
            help="Number of characters per chunk"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=config.document.chunk_overlap,
            help="Number of characters to overlap between chunks"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=0,
            help="Model to use for generating embeddings"
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value=config.vector_store.collection_name,
            help="Name of the vector store collection"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            max_workers = st.slider(
                "Concurrent Workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of concurrent workers for processing"
            )
            
            max_file_size = st.slider(
                "Max File Size (MB)",
                min_value=1,
                max_value=100,
                value=config.document.max_file_size_mb,
                help="Maximum file size allowed"
            )

    # File upload section
    uploaded_files = st.file_uploader(
        "📁 Upload your documents",
        accept_multiple_files=True,
        type=[ext.lstrip('.') for ext in DocumentLoaderFactory.get_supported_extensions()],
        help=f"Maximum file size: {max_file_size} MB per file"
    )
    
    # Display file information
    if uploaded_files:
        st.subheader(f"📋 Selected Files ({len(uploaded_files)})")
        
        total_size = 0
        file_types = {}
        
        for file in uploaded_files:
            file_size = len(file.getvalue())
            total_size += file_size
            file_ext = Path(file.name).suffix.lower()
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
            
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"📄 {file.name}")
            col2.write(f"{file_size / 1024:.1f} KB")
            col3.write(file_ext.upper())
        
        # Summary
        st.info(f"**Total:** {len(uploaded_files)} files, {total_size / 1024 / 1024:.2f} MB")
        
        # File type breakdown
        if len(file_types) > 1:
            st.write("**File types:**", " | ".join([f"{ext}: {count}" for ext, count in file_types.items()]))

    if uploaded_files and st.button("🚀 Start Ingestion", type="primary"):
        try:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            # Save uploaded files and get their paths
            status_text.text("💾 Saving uploaded files...")
            progress_bar.progress(10)
            
            temp_paths = []
            failed_files = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = save_uploaded_file(uploaded_file)
                if temp_path:
                    temp_paths.append(temp_path)
                else:
                    failed_files.append(uploaded_file.name)
                
                progress_bar.progress(10 + (i + 1) * 20 // len(uploaded_files))
            
            if not temp_paths:
                st.error("No files could be saved successfully.")
                return
            
            # Initialize ingester with configured parameters
            status_text.text("⚙️ Initializing ingestion pipeline...")
            progress_bar.progress(30)
            
            ingester = DocumentIngester(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                persist_directory=str(DB_PATH),
                max_workers=max_workers
            )
            
            # Perform health check
            status_text.text("🔍 Performing health check...")
            progress_bar.progress(40)
            
            health = ingester.health_check()
            if health["status"] != "healthy":
                st.warning("⚠️ Health check detected issues - proceeding with caution")
                with st.expander("Health Check Details"):
                    st.json(health)
            
            # Process documents
            status_text.text("🔄 Processing documents...")
            progress_bar.progress(50)
            
            vectorstore = ingester.ingest_files(temp_paths, collection_name)
            
            # Update progress
            progress_bar.progress(90)
            
            # Get final stats
            status_text.text("📊 Generating statistics...")
            stats = ingester.get_ingestion_stats()
            
            processing_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text("✅ Ingestion completed successfully!")
            
            # Display comprehensive summary
            st.success("🎉 Document ingestion completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Processing Summary")
                st.metric("Files Processed", len(temp_paths))
                st.metric("Processing Time", f"{processing_time:.1f}s")
                st.metric("Collection Name", collection_name)
                
                if failed_files:
                    st.error(f"⚠️ Failed files: {', '.join(failed_files)}")
            
            with col2:
                st.subheader("⚙️ Configuration Used")
                st.write(f"**Chunk Size:** {chunk_size}")
                st.write(f"**Chunk Overlap:** {chunk_overlap}")
                st.write(f"**Embedding Model:** {embedding_model}")
                st.write(f"**Max Workers:** {max_workers}")
            
            # Display detailed stats
            if stats and "collections" in stats:
                with st.expander("📊 Detailed Statistics", expanded=True):
                    st.json(stats)
            
            # Cleanup temporary files
            for temp_path in temp_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"❌ Error during ingestion: {str(e)}")
            
            # Show error details for debugging
            if st.checkbox("Show error details"):
                st.exception(e)

def render_chatbot_tab():
    st.header("🤖 Document Q&A")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Chatbot Configuration")
        
        # Get available collections with enhanced info
        available_collections = RAGChatbot.get_available_collections()
        if not available_collections:
            st.warning("⚠️ No document collections found. Please ingest some documents first.")
            return
        
        # Display collection information
        st.subheader("📚 Available Collections")
        collection_options = []
        for collection in available_collections:
            name = collection['name']
            count = collection['count']
            file_types = collection.get('file_types', [])
            
            # Create display name with info
            display_name = f"{name} ({count} docs"
            if file_types:
                display_name += f", {len(file_types)} types"
            display_name += ")"
            
            collection_options.append((name, display_name))
            
            # Show collection details
            with st.expander(f"📋 {name}", expanded=False):
                st.write(f"**Documents:** {count}")
                if file_types:
                    st.write(f"**File Types:** {', '.join(file_types)}")
                
                sample_sources = collection.get('sample_sources', [])
                if sample_sources:
                    st.write("**Sample Files:**")
                    for source in sample_sources[:3]:
                        st.write(f"• {Path(source).name}")
        
        # Collection selection
        selected_collection = st.selectbox(
            "Select Document Collection",
            options=[opt[0] for opt in collection_options],
            format_func=lambda x: next((opt[1] for opt in collection_options if opt[0] == x), x),
            help="Choose which document collection to query"
        )
        
        # LLM Provider and Model configuration
        from config import get_available_llm_providers
        
        providers_info = get_available_llm_providers()
        provider_names = list(providers_info.keys())
        
        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_names,
            index=0,  # Default to first provider (groq)
            help="Choose your LLM provider. Each provider requires different API keys."
        )
        
        provider_info = providers_info[selected_provider]
        
        # Show provider information
        with st.expander(f"ℹ️ {provider_info['name']} Configuration", expanded=False):
            st.write(f"**Description:** {provider_info['description']}")
            
            if provider_info.get('requires_api_key', True):
                env_var = provider_info['env_var']
                api_key_set = bool(os.getenv(env_var))
                
                if api_key_set:
                    st.success(f"✅ API key configured ({env_var})")
                else:
                    st.error(f"❌ API key not set. Please set {env_var} environment variable.")
                    
            if selected_provider == 'ollama':
                ollama_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
                st.info(f"**API Base:** {ollama_base}")
                st.write("Make sure Ollama is running locally or set OLLAMA_API_BASE to your Ollama server.")
        
        # Model selection based on provider
        available_models = provider_info.get('models', [])
        if available_models:
            model_name = st.selectbox(
                f"{provider_info['name']} Model",
                options=available_models,
                help=f"Available models for {provider_info['name']}"
            )
        else:
            model_name = st.text_input(
                "Custom Model Name",
                value="llama-3.1-8b-instant",
                help="Enter the model name for your provider"
            )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2", 
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1,
            help="Model used for document embeddings - must match ingestion model"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider(
                "Response Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.llm.temperature,
                step=0.1,
                help="Controls randomness in responses (0=deterministic, 1=creative)"
            )
            
            max_results = st.slider(
                "Max Retrieved Documents",
                min_value=1,
                max_value=10,
                value=config.vector_store.max_results,
                help="Number of document chunks to retrieve for context"
            )
        
        # Initialize/Reinitialize button
        init_button_text = "🚀 Initialize Chatbot"
        if "chatbot" in st.session_state and st.session_state.chatbot is not None:
            init_button_text = "🔄 Reinitialize Chatbot"
        
        if st.button(init_button_text, type="primary"):
            with st.spinner("🔄 Initializing chatbot..."):
                try:
                    st.session_state.chatbot = RAGChatbot(
                        collection_name=selected_collection,
                        model_name=model_name,
                        embedding_model=embedding_model,
                        llm_provider=selected_provider
                    )
                    
                    # Perform health check
                    health = st.session_state.chatbot.health_check()
                    
                    if health["status"] == "healthy":
                        st.success("✅ Chatbot initialized successfully!")
                    else:
                        st.warning(f"⚠️ Chatbot initialized with status: {health['status']}")
                        
                    # Show health details
                    with st.expander("🏥 Health Check Details", expanded=False):
                        st.json(health)
                        
                except Exception as e:
                    st.error(f"❌ Error initializing chatbot: {str(e)}")
                    
                    # Show detailed error info
                    if isinstance(e, (VectorStoreError, ConfigurationError)):
                        st.error("💡 **Troubleshooting tips:**")
                        if "does not exist" in str(e):
                            st.error("• Make sure you've ingested documents with the correct collection name")
                        elif "empty" in str(e):
                            st.error("• The collection exists but has no documents - try ingesting files")
                        elif "API_KEY" in str(e) or "api_key" in str(e):
                            st.error("• Set your LLM provider API key environment variable")
                            st.error("• Check the configuration for your selected LLM provider")
    
    # Main chat interface
    if "chatbot" not in st.session_state or st.session_state.chatbot is None:
        st.info("👈 Please initialize the chatbot using the sidebar configuration.")
        
        # Show some helpful examples
        st.subheader("💡 Example Questions")
        examples = [
            "What types of documents are in the collection?",
            "Summarize the main topics covered in the documents",
            "What are the key features mentioned?",
            "Can you provide statistics or metrics from the data?",
            "What file formats are supported?"
        ]
        
        for example in examples:
            if st.button(f"💭 {example}", key=f"example_{hash(example)}"):
                st.info("Initialize the chatbot first to ask questions!")
        
        return
    
    # Chat history display
    for human, ai in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("assistant"):
            st.write(ai)
    
    # Chat input with enhanced features
    if question := st.chat_input("💬 Ask a question about your documents..."):
        # Validate input
        if len(question.strip()) < 3:
            st.warning("Please provide a more detailed question.")
            return
        
        # Display user message
        with st.chat_message("human"):
            st.write(question)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    start_time = time.time()
                    response = st.session_state.chatbot.get_response(
                        question,
                        st.session_state.chat_history
                    )
                    response_time = time.time() - start_time
                    
                    st.write(response)
                    
                    # Show response time
                    st.caption(f"⏱️ Response generated in {response_time:.1f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    response = "I encountered an error while processing your question. Please try again."
        
        # Update chat history
        st.session_state.chat_history.append((question, response))
        
        # Limit chat history to prevent memory issues
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
    
    # Chat management
    if st.session_state.chat_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 Chat Management")
        
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export chat option
        if st.sidebar.button("📥 Export Chat"):
            chat_export = []
            for human, ai in st.session_state.chat_history:
                chat_export.append({"question": human, "answer": ai})
            
            import json
            chat_json = json.dumps(chat_export, indent=2)
            st.sidebar.download_button(
                label="💾 Download Chat History",
                data=chat_json,
                file_name=f"chat_history_{int(time.time())}.json",
                mime="application/json"
            )

def main():
    st.title("🤖 RAG Document Assistant")
    st.caption("Enhanced with multi-format document support and production-ready features")
    
    initialize_session_state()
    
    # Create tabs with enhanced names
    tab1, tab2, tab3 = st.tabs(["📚 Document Ingestion", "🤖 Document Q&A", "🔧 System Status"])
    
    with tab1:
        render_ingestion_tab()
    
    with tab2:
        render_chatbot_tab()
    
    with tab3:
        render_system_status_tab()
    
    # Enhanced sidebar with system information
    with st.sidebar:
        st.markdown("---")
        st.subheader("📋 System Information")
        
        # Configuration summary
        with st.expander("⚙️ Configuration", expanded=False):
            st.write(f"**Environment:** {config.environment}")
            st.write(f"**Debug Mode:** {config.debug}")
            st.write(f"**DB Path:** {config.vector_store.persist_directory}")
            st.write(f"**Max File Size:** {config.document.max_file_size_mb} MB")
        
        # Supported formats
        with st.expander("📄 Supported Formats", expanded=False):
            extensions = DocumentLoaderFactory.get_supported_extensions()
            for ext in extensions:
                st.write(f"• `{ext}`")
        
        # Quick stats
        try:
            collections = RAGChatbot.get_available_collections()
            if collections:
                total_docs = sum(c['count'] for c in collections)
                st.metric("Total Documents", total_docs)
                st.metric("Collections", len(collections))
        except:
            pass

def render_system_status_tab():
    """Render system status and health information."""
    st.header("🔧 System Status & Health")
    
    # Overall system health
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Health Check")
        
        if st.button("🔍 Run Health Check", type="primary"):
            with st.spinner("Performing health check..."):
                try:
                    # Try to create a temporary ingester for health check
                    ingester = DocumentIngester(
                        persist_directory=str(DB_PATH)
                    )
                    health = ingester.health_check()
                    
                    # Display health status
                    status_color = {
                        "healthy": "🟢",
                        "degraded": "🟡", 
                        "unhealthy": "🔴"
                    }
                    
                    st.write(f"{status_color.get(health['status'], '⚪')} **Status:** {health['status'].upper()}")
                    
                    # Display individual checks
                    for check_name, check_result in health.get('checks', {}).items():
                        status_icon = "✅" if check_result.get('status') == 'ok' else "❌"
                        st.write(f"{status_icon} **{check_name.replace('_', ' ').title()}**")
                        
                        if check_result.get('status') != 'ok':
                            st.error(f"Error: {check_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Health check failed: {str(e)}")
    
    with col2:
        st.subheader("📊 Collections Overview")
        
        try:
            collections = RAGChatbot.get_available_collections()
            
            if collections:
                for collection in collections:
                    with st.container():
                        st.write(f"**📚 {collection['name']}**")
                        st.write(f"Documents: {collection['count']}")
                        
                        if collection.get('file_types'):
                            st.write(f"Types: {', '.join(collection['file_types'])}")
                        
                        st.markdown("---")
            else:
                st.info("No collections found. Ingest some documents to see them here.")
                
        except Exception as e:
            st.error(f"Error loading collections: {str(e)}")
    
    # Configuration details
    st.subheader("⚙️ Current Configuration")
    
    config_data = {
        "Document Processing": {
            "Chunk Size": config.document.chunk_size,
            "Chunk Overlap": config.document.chunk_overlap,
            "Max File Size (MB)": config.document.max_file_size_mb,
            "Supported Extensions": len(config.document.allowed_extensions)
        },
        "Embeddings": {
            "Model": config.embedding.model_name,
            "Batch Size": config.embedding.batch_size,
            "Max Retries": config.embedding.max_retries
        },
        "Vector Store": {
            "Directory": config.vector_store.persist_directory,
            "Collection": config.vector_store.collection_name,
            "Similarity Threshold": config.vector_store.similarity_threshold
        },
        "LLM": {
            "Model": config.llm.model_name,
            "Temperature": config.llm.temperature,
            "Max Tokens": config.llm.max_tokens
        }
    }
    
    for section, settings in config_data.items():
        with st.expander(f"📋 {section}", expanded=False):
            for key, value in settings.items():
                st.write(f"**{key}:** {value}")
    
    # Environment variables
    st.subheader("🌍 Environment")
    
    env_vars = [
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "OLLAMA_API_BASE",
        "CHUNK_SIZE", 
        "CHUNK_OVERLAP",
        "EMBEDDING_MODEL",
        "VECTOR_DB_PATH",
        "LOG_LEVEL"
    ]
    
    env_status = {}
    for var in env_vars:
        value = os.environ.get(var)
        env_status[var] = "✅ Set" if value else "❌ Not set"
    
    for var, status in env_status.items():
        st.write(f"**{var}:** {status}")
    
    # Disk usage
    st.subheader("💾 Storage Information")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(config.vector_store.persist_directory)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Space", f"{total / (1024**3):.1f} GB")
        col2.metric("Used Space", f"{used / (1024**3):.1f} GB") 
        col3.metric("Free Space", f"{free / (1024**3):.1f} GB")
        
        # Usage percentage
        usage_pct = (used / total) * 100
        st.progress(usage_pct / 100)
        st.caption(f"Disk usage: {usage_pct:.1f}%")
        
    except Exception as e:
        st.error(f"Could not get disk usage: {str(e)}")

if __name__ == "__main__":
    main() 
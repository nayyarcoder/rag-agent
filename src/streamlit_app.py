import streamlit as st
import tempfile
import os
from pathlib import Path
from ingestion import DocumentIngester
from chatbot import RAGChatbot

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Add DB_PATH constant
DB_PATH = Path("db")
if not DB_PATH.exists():
    DB_PATH.mkdir(parents=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
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

    # Sidebar for configuration
    with st.sidebar:
        st.header("Ingestion Configuration")
        
        chunk_size = st.slider(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=1000,
            help="Number of characters per chunk"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=200,
            help="Number of characters to overlap between chunks"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            help="Model to use for generating embeddings"
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value="default",
            help="Name of the vector store collection"
        )

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "doc", "docx"]
    )

    if uploaded_files and st.button("Start Ingestion"):
        try:
            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save uploaded files and get their paths
            status_text.text("Saving uploaded files...")
            temp_paths = []
            for uploaded_file in uploaded_files:
                temp_path = save_uploaded_file(uploaded_file)
                if temp_path:
                    temp_paths.append(temp_path)
            
            # Initialize ingester with configured parameters
            status_text.text("Initializing ingestion pipeline...")
            ingester = DocumentIngester(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                persist_directory=str(DB_PATH)
            )
            
            # Process documents
            status_text.text("Processing documents...")
            vectorstore = ingester.ingest_files(temp_paths, collection_name)
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("✅ Ingestion completed successfully!")
            
            # Display summary
            st.success(f"""
            Successfully processed {len(uploaded_files)} documents:
            - Collection name: {collection_name}
            - Chunk size: {chunk_size}
            - Chunk overlap: {chunk_overlap}
            - Embedding model: {embedding_model}
            """)
            
            # Cleanup temporary files
            for temp_path in temp_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Error during ingestion: {str(e)}")

def render_chatbot_tab():
    st.header("🤖 Document Q&A")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Chatbot Configuration")
        
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
            options=["llama-3.1-8b-instant", "llama3-8b-8192"],
            help="Groq model to use"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            help="Model to use for generating embeddings - must match the model used during ingestion"
        )
        
        if st.button("Initialize Chatbot"):
            with st.spinner("Initializing chatbot..."):
                try:
                    st.session_state.chatbot = RAGChatbot(
                        collection_name=collection_name,
                        model_name=model_name,
                        embedding_model=embedding_model
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

def main():
    st.title("📚 RAG Document Assistant")
    
    initialize_session_state()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Document Ingestion", "Document Q&A"])
    
    with tab1:
        render_ingestion_tab()
    
    with tab2:
        render_chatbot_tab()
    
    # Display supported file types in sidebar
    st.sidebar.markdown("""
    ### Supported File Types
    - PDF (.pdf)
    - Text (.txt)
    - Word (.doc, .docx)
    """)

if __name__ == "__main__":
    main() 
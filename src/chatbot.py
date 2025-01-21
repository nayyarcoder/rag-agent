import os
from dotenv import load_dotenv
import streamlit as st
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class RAGChatbot:
    def __init__(
        self,
        persist_directory: str = "db",
        collection_name: str = "default",
        model_name: str = "llama-3.1-8b-instant",
        embedding_model: str = "all-mpnet-base-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing RAGChatbot with collection: {collection_name}")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Verify collection exists and get stats
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            logger.info(f"Collection '{collection_name}' found with {count} documents")
            if count == 0:
                raise ValueError(f"Collection '{collection_name}' exists but is empty")
        except Exception as e:
            logger.error(f"Error accessing collection: {str(e)}")
            raise
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Test retrieval functionality
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            logger.info("Testing retriever functionality...")
            test_results = retriever.get_relevant_documents("test query")
            logger.info(f"Retriever test returned {len(test_results)} documents")
        except Exception as e:
            logger.error(f"Error testing retriever: {str(e)}")
            raise
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            temperature=0.7,
            model_name=model_name,
            api_key=os.environ["GROQ_API_KEY"]
        )
        
        # Create the RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided document context.
            Your responses should be:
            1. Based ONLY on the information found in the provided context
            2. Clear and concise
            3. If the answer cannot be found in the context, say "I cannot find information about that in the provided documents."
            4. If you need more context to provide a complete answer, say so.
            
            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create the RAG chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            combine_docs_chain_kwargs={
                "prompt": self.prompt
            },
            return_source_documents=True,
            memory=None,
            get_chat_history=lambda h: h
        )
        logger.info("RAGChatbot initialization completed successfully")

    @staticmethod
    def get_available_collections(persist_directory: str = "db") -> List[str]:
        """Get a list of available collections in the Chroma database."""
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            collections = client.list_collections()
            collection_info = []
            for collection in collections:
                count = collection.count()
                collection_info.append({
                    'name': collection.name,
                    'count': count
                })
            logger.info(f"Found collections: {collection_info}")
            return [c['name'] for c in collection_info if c['count'] > 0]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
        
    def get_response(self, question: str, chat_history: List[tuple]) -> str:
        """Get a response from the chatbot."""
        try:
            logger.info(f"\nProcessing question: {question}")
            logger.info(f"Current collection: {self.collection_name}")
            
            # Convert chat history to Message objects
            messages = []
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=str(human_msg)))
                messages.append(AIMessage(content=str(ai_msg)))
            
            # Get response from the chain
            logger.info("Retrieving documents from vector store...")
            response = self.chain.invoke({
                "question": question,
                "chat_history": messages
            })
            
            # Log retrieved chunks
            source_docs = response.get("source_documents", [])
            logger.info(f"Retrieved {len(source_docs)} chunks")
            
            # Create an expandable section in Streamlit to show chunks
            with st.expander("View Retrieved Chunks", expanded=False):
                st.markdown("### Retrieved Document Chunks")
                if not source_docs:
                    st.warning("No chunks were retrieved for this query.")
                    # Display collection stats
                    collection = self.client.get_collection(self.collection_name)
                    st.info(f"Collection '{self.collection_name}' has {collection.count()} total documents")
                else:
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        if hasattr(doc.metadata, 'source'):
                            st.caption(f"Source: {doc.metadata.source}")
                        st.markdown("---")
            
            # Check if we got any source documents
            if not source_docs:
                return "I cannot find any relevant information in the documents to answer your question. Please try rephrasing your question or ask about a different topic."
            
            return response["answer"]
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return f"Error: {str(e)}"

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

def main():
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
            options=["llama-3.1-8b-instant", "llama3-8b-8192"],
            help="Groq model to use"
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
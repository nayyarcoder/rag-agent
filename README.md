# RAG Document Assistant 🤖

A Retrieval Augmented Generation (RAG) chatbot that can understand and answer questions about your documents. Built with LangChain, ChromaDB, and Streamlit.

## Features

- 📄 Support for multiple document formats (PDF, TXT, DOCX)
- 💾 Document ingestion and vector storage using ChromaDB
- 🔍 Semantic search and retrieval
- 💬 Interactive chat interface
- 🚀 Built with modern LLM technologies

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/rag-agent.git
    cd rag-agent
    ```

1. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On Unix or MacOS
    source .venv/bin/activate
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your environment variables:

    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Project Structure

```text
rag-agent/
├── data/           # Store your documents here
├── db/             # ChromaDB vector store
├── src/
│   ├── ingestion.py         # Document ingestion logic
│   ├── chatbot.py           # RAG chatbot implementation
│   ├── streamlit_app.py     # Streamlit web interface
│   └── test_*.py           # Test files
├── requirements.txt
└── README.md
```

## Usage

1. Start the Streamlit application:

    ```bash
    streamlit run src/streamlit_app.py
    ```

2. Upload your documents in the "Document Ingestion" tab.

3. Switch to the "Chat" tab to start asking questions about your documents.

## Dependencies

- langchain & langchain-community: For LLM interactions and RAG implementation
- chromadb: Vector store for document embeddings
- sentence-transformers: For text embeddings
- streamlit: Web interface
- Various document processing libraries (unstructured, python-docx, pypdf)

## Development

To run tests:

```bash
pytest src/test_*.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

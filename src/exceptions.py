"""Custom exceptions for the RAG agent application."""

class RAGAgentException(Exception):
    """Base exception for RAG agent errors."""
    pass

class DocumentLoadError(RAGAgentException):
    """Raised when a document cannot be loaded."""
    def __init__(self, file_path: str, message: str):
        self.file_path = file_path
        super().__init__(f"Failed to load document '{file_path}': {message}")

class UnsupportedDocumentTypeError(RAGAgentException):
    """Raised when a document type is not supported."""
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type
        super().__init__(f"Unsupported document type '{file_type}' for file '{file_path}'")

class DocumentProcessingError(RAGAgentException):
    """Raised when document processing fails."""
    def __init__(self, message: str, file_path: str = None):
        self.file_path = file_path
        super().__init__(f"Document processing error: {message}")

class VectorStoreError(RAGAgentException):
    """Raised when vector store operations fail."""
    pass

class EmbeddingError(RAGAgentException):
    """Raised when embedding generation fails."""
    pass

class ConfigurationError(RAGAgentException):
    """Raised when configuration is invalid."""
    pass

class FileSizeError(RAGAgentException):
    """Raised when file size exceeds limits."""
    def __init__(self, file_path: str, size: int, max_size: int):
        self.file_path = file_path
        self.size = size
        self.max_size = max_size
        super().__init__(f"File '{file_path}' size {size} bytes exceeds maximum allowed size {max_size} bytes")

class FileValidationError(RAGAgentException):
    """Raised when file validation fails."""
    def __init__(self, file_path: str, message: str):
        self.file_path = file_path
        super().__init__(f"File validation failed for '{file_path}': {message}")
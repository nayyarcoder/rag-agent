"""Configuration management for the RAG agent."""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50  # Maximum file size in MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.csv', '.json', '.md'
    ])
    encoding: str = 'utf-8'

@dataclass 
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_retries: int = 3
    timeout_seconds: int = 30

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "db"
    collection_name: str = "documents"
    similarity_threshold: float = 0.7
    max_results: int = 10

@dataclass
class LLMConfig:
    """Configuration for LLM."""
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    scan_uploads: bool = True
    allowed_mime_types: List[str] = field(default_factory=lambda: [
        'text/plain', 'application/pdf', 'text/html', 'text/csv',
        'application/json', 'text/markdown',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword'
    ])
    max_concurrent_uploads: int = 10

@dataclass
class AppConfig:
    """Main application configuration."""
    document: DocumentConfig = field(default_factory=DocumentConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment-specific settings
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Document config
        config.document.chunk_size = int(os.getenv('CHUNK_SIZE', config.document.chunk_size))
        config.document.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', config.document.chunk_overlap))
        config.document.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', config.document.max_file_size_mb))
        
        # Embedding config
        config.embedding.model_name = os.getenv('EMBEDDING_MODEL', config.embedding.model_name)
        config.embedding.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', config.embedding.batch_size))
        
        # Vector store config
        config.vector_store.persist_directory = os.getenv('VECTOR_DB_PATH', config.vector_store.persist_directory)
        config.vector_store.collection_name = os.getenv('COLLECTION_NAME', config.vector_store.collection_name)
        
        # LLM config
        config.llm.model_name = os.getenv('LLM_MODEL', config.llm.model_name)
        config.llm.temperature = float(os.getenv('LLM_TEMPERATURE', config.llm.temperature))
        
        # Logging config
        config.logging.level = os.getenv('LOG_LEVEL', config.logging.level)
        config.logging.file_path = os.getenv('LOG_FILE_PATH', config.logging.file_path)
        
        # App config
        config.environment = os.getenv('ENVIRONMENT', config.environment)
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        return config
    
    def validate(self) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate chunk size
        if self.document.chunk_size <= 0:
            errors.append("chunk_size must be greater than 0")
        
        if self.document.chunk_overlap < 0:
            errors.append("chunk_overlap must be non-negative")
        
        if self.document.chunk_overlap >= self.document.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        # Validate file size
        if self.document.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be greater than 0")
        
        # Validate paths
        if not self.vector_store.persist_directory:
            errors.append("persist_directory cannot be empty")
        
        # Validate LLM settings
        if not (0 <= self.llm.temperature <= 2):
            errors.append("temperature must be between 0 and 2")
        
        if self.llm.max_tokens <= 0:
            errors.append("max_tokens must be greater than 0")
        
        # Validate logging
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_levels:
            errors.append(f"log_level must be one of {valid_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        import logging.handlers
        
        # Create logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.logging.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if self.logging.file_path:
            log_path = Path(self.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

# Global configuration instance
config = AppConfig.from_env()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update global configuration."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.validate()
"""Enhanced document loaders for various file types."""

import os
import json
import csv
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from exceptions import (
    DocumentLoadError, 
    UnsupportedDocumentTypeError,
    FileSizeError,
    FileValidationError
)
from config import get_config

logger = logging.getLogger(__name__)

class BaseDocumentLoader(ABC):
    """Base class for document loaders."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.config = get_config()
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from file."""
        pass
    
    def validate_file(self) -> None:
        """Validate file before loading."""
        file_path = Path(self.file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileValidationError(self.file_path, "File does not exist")
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.config.document.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise FileSizeError(self.file_path, file_size, max_size)
        
        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext not in self.config.document.allowed_extensions:
            raise UnsupportedDocumentTypeError(self.file_path, file_ext)
        
        # Check MIME type if security scanning is enabled
        if self.config.security.scan_uploads:
            mime_type, _ = mimetypes.guess_type(self.file_path)
            if mime_type and mime_type not in self.config.security.allowed_mime_types:
                raise FileValidationError(self.file_path, f"MIME type '{mime_type}' not allowed")

class EnhancedPDFLoader(BaseDocumentLoader):
    """Enhanced PDF loader with better error handling."""
    
    def load(self) -> List[Document]:
        """Load PDF documents."""
        self.validate_file()
        
        try:
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            
            if not documents:
                raise DocumentLoadError(self.file_path, "No content extracted from PDF")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF: {self.file_path}")
            return documents
            
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"PDF loading failed: {str(e)}")

class EnhancedTextLoader(BaseDocumentLoader):
    """Enhanced text loader with encoding detection."""
    
    def load(self) -> List[Document]:
        """Load text documents."""
        self.validate_file()
        
        try:
            # Try primary encoding first
            encoding = self.config.document.encoding
            try:
                with open(self.file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to common encodings
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(self.file_path, 'r', encoding=fallback_encoding) as f:
                            content = f.read()
                        encoding = fallback_encoding
                        logger.warning(f"Used fallback encoding {encoding} for {self.file_path}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise DocumentLoadError(self.file_path, "Could not decode file with any supported encoding")
            
            if not content.strip():
                raise DocumentLoadError(self.file_path, "File is empty or contains only whitespace")
            
            document = Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "encoding": encoding,
                    "file_size": os.path.getsize(self.file_path)
                }
            )
            
            logger.info(f"Successfully loaded text file: {self.file_path}")
            return [document]
            
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"Text loading failed: {str(e)}")

class EnhancedDocxLoader(BaseDocumentLoader):
    """Enhanced DOCX loader."""
    
    def load(self) -> List[Document]:
        """Load DOCX documents."""
        self.validate_file()
        
        try:
            loader = Docx2txtLoader(self.file_path)
            documents = loader.load()
            
            if not documents:
                raise DocumentLoadError(self.file_path, "No content extracted from DOCX")
            
            # Add additional metadata
            for doc in documents:
                doc.metadata.update({
                    "file_size": os.path.getsize(self.file_path),
                    "file_type": "docx"
                })
            
            logger.info(f"Successfully loaded DOCX file: {self.file_path}")
            return documents
            
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"DOCX loading failed: {str(e)}")

class HTMLLoader(BaseDocumentLoader):
    """HTML document loader."""
    
    def load(self) -> List[Document]:
        """Load HTML documents."""
        self.validate_file()
        
        try:
            from bs4 import BeautifulSoup
            
            with open(self.file_path, 'r', encoding=self.config.document.encoding) as f:
                content = f.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            if not text_content.strip():
                raise DocumentLoadError(self.file_path, "No text content found in HTML")
            
            # Extract title if available
            title = soup.find('title')
            title_text = title.get_text().strip() if title else Path(self.file_path).stem
            
            document = Document(
                page_content=text_content,
                metadata={
                    "source": self.file_path,
                    "title": title_text,
                    "file_type": "html",
                    "file_size": os.path.getsize(self.file_path)
                }
            )
            
            logger.info(f"Successfully loaded HTML file: {self.file_path}")
            return [document]
            
        except ImportError:
            raise DocumentLoadError(self.file_path, "BeautifulSoup4 is required for HTML loading")
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"HTML loading failed: {str(e)}")

class CSVLoader(BaseDocumentLoader):
    """CSV document loader."""
    
    def load(self) -> List[Document]:
        """Load CSV documents."""
        self.validate_file()
        
        try:
            documents = []
            
            with open(self.file_path, 'r', encoding=self.config.document.encoding) as f:
                # Detect delimiter
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    # Convert row to text
                    row_text = ' | '.join(f"{k}: {v}" for k, v in row.items() if v)
                    
                    if row_text.strip():
                        document = Document(
                            page_content=row_text,
                            metadata={
                                "source": self.file_path,
                                "row_number": row_num,
                                "file_type": "csv",
                                "columns": list(row.keys())
                            }
                        )
                        documents.append(document)
            
            if not documents:
                raise DocumentLoadError(self.file_path, "No valid rows found in CSV")
            
            logger.info(f"Successfully loaded {len(documents)} rows from CSV: {self.file_path}")
            return documents
            
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"CSV loading failed: {str(e)}")

class JSONLoader(BaseDocumentLoader):
    """JSON document loader."""
    
    def load(self) -> List[Document]:
        """Load JSON documents."""
        self.validate_file()
        
        try:
            with open(self.file_path, 'r', encoding=self.config.document.encoding) as f:
                data = json.load(f)
            
            documents = []
            
            def extract_text_from_json(obj, path=""):
                """Recursively extract text from JSON object."""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        extract_text_from_json(value, new_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_path = f"{path}[{i}]"
                        extract_text_from_json(item, new_path)
                elif isinstance(obj, (str, int, float, bool)):
                    if str(obj).strip():
                        content = f"{path}: {obj}" if path else str(obj)
                        document = Document(
                            page_content=content,
                            metadata={
                                "source": self.file_path,
                                "json_path": path,
                                "file_type": "json"
                            }
                        )
                        documents.append(document)
            
            extract_text_from_json(data)
            
            if not documents:
                raise DocumentLoadError(self.file_path, "No extractable content found in JSON")
            
            logger.info(f"Successfully loaded {len(documents)} elements from JSON: {self.file_path}")
            return documents
            
        except json.JSONDecodeError as e:
            raise DocumentLoadError(self.file_path, f"Invalid JSON format: {str(e)}")
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"JSON loading failed: {str(e)}")

class MarkdownLoader(BaseDocumentLoader):
    """Markdown document loader."""
    
    def load(self) -> List[Document]:
        """Load Markdown documents."""
        self.validate_file()
        
        try:
            with open(self.file_path, 'r', encoding=self.config.document.encoding) as f:
                content = f.read()
            
            if not content.strip():
                raise DocumentLoadError(self.file_path, "Markdown file is empty")
            
            # Basic markdown processing - convert to plain text
            # Remove markdown syntax while preserving structure
            lines = content.split('\n')
            processed_lines = []
            
            for line in lines:
                # Remove markdown headers
                line = line.lstrip('#').strip()
                # Remove bold/italic markers
                line = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
                # Remove inline code
                line = line.replace('`', '')
                # Keep line if not empty
                if line.strip():
                    processed_lines.append(line)
            
            processed_content = '\n'.join(processed_lines)
            
            document = Document(
                page_content=processed_content,
                metadata={
                    "source": self.file_path,
                    "file_type": "markdown",
                    "file_size": os.path.getsize(self.file_path)
                }
            )
            
            logger.info(f"Successfully loaded Markdown file: {self.file_path}")
            return [document]
            
        except Exception as e:
            if isinstance(e, (DocumentLoadError, FileValidationError)):
                raise
            raise DocumentLoadError(self.file_path, f"Markdown loading failed: {str(e)}")

class DocumentLoaderFactory:
    """Factory for creating document loaders."""
    
    _loaders = {
        '.pdf': EnhancedPDFLoader,
        '.txt': EnhancedTextLoader,
        '.docx': EnhancedDocxLoader,
        '.doc': EnhancedDocxLoader,
        '.html': HTMLLoader,
        '.htm': HTMLLoader,
        '.csv': CSVLoader,
        '.json': JSONLoader,
        '.md': MarkdownLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: str) -> BaseDocumentLoader:
        """Create appropriate loader for file type."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in cls._loaders:
            raise UnsupportedDocumentTypeError(file_path, file_ext)
        
        loader_class = cls._loaders[file_ext]
        return loader_class(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._loaders.keys())
    
    @classmethod
    def register_loader(cls, extension: str, loader_class: type) -> None:
        """Register a new loader for a file extension."""
        cls._loaders[extension] = loader_class
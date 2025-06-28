# Sample Markdown Document

This is a **sample markdown document** for testing the enhanced RAG agent.

## Features

The enhanced RAG agent now supports:

- PDF documents
- Text files  
- Word documents (DOCX/DOC)
- HTML files
- CSV files
- JSON files
- Markdown files

## Benefits

1. **Robust error handling** - Better exception management
2. **Multiple document types** - Support for various formats
3. **Production ready** - Enhanced logging and monitoring
4. **Scalable** - Concurrent processing support

### Code Example

```python
from document_loaders import DocumentLoaderFactory

loader = DocumentLoaderFactory.create_loader("document.md")
documents = loader.load()
```

This demonstrates the *factory pattern* for document loading.

## Conclusion

The enhanced system provides a solid foundation for production RAG applications.
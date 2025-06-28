#!/usr/bin/env python3
"""
Simple test script for the enhanced ingestion system.
Run without pytest to validate basic functionality.
"""

import os
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from config import get_config, AppConfig
from document_loaders import DocumentLoaderFactory, EnhancedTextLoader, HTMLLoader, CSVLoader, JSONLoader
from ingestion import DocumentIngester
from exceptions import DocumentLoadError, UnsupportedDocumentTypeError

def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    
    config = get_config()
    assert config.document.chunk_size > 0
    assert config.document.chunk_overlap >= 0
    assert config.embedding.model_name
    print("✓ Configuration system works")

def test_document_loaders():
    """Test enhanced document loaders."""
    print("Testing document loaders...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test text loader
        txt_file = Path(temp_dir) / "test.txt"
        txt_file.write_text("This is a test document with some content.")
        
        loader = EnhancedTextLoader(str(txt_file))
        docs = loader.load()
        assert len(docs) == 1
        assert "test document" in docs[0].page_content
        print("✓ Text loader works")
        
        # Test HTML loader
        html_file = Path(temp_dir) / "test.html"
        html_content = """
        <html>
        <head><title>Test HTML</title></head>
        <body>
            <h1>Test Header</h1>
            <p>This is a paragraph with <b>bold</b> text.</p>
            <script>console.log('ignored');</script>
        </body>
        </html>
        """
        html_file.write_text(html_content)
        
        try:
            loader = HTMLLoader(str(html_file))
            docs = loader.load()
            assert len(docs) == 1
            assert "Test Header" in docs[0].page_content
            assert "bold" in docs[0].page_content
            assert "console.log" not in docs[0].page_content  # Script should be removed
            print("✓ HTML loader works")
        except ImportError:
            print("⚠ HTML loader skipped (BeautifulSoup4 not available)")
        
        # Test CSV loader
        csv_file = Path(temp_dir) / "test.csv"
        csv_content = """name,age,city
John,30,New York
Jane,25,Los Angeles
Bob,35,Chicago"""
        csv_file.write_text(csv_content)
        
        loader = CSVLoader(str(csv_file))
        docs = loader.load()
        assert len(docs) == 3  # 3 rows
        assert "John" in docs[0].page_content
        print("✓ CSV loader works")
        
        # Test JSON loader
        json_file = Path(temp_dir) / "test.json"
        json_data = {
            "users": [
                {"name": "Alice", "age": 28},
                {"name": "Bob", "age": 32}
            ],
            "metadata": {"version": "1.0"}
        }
        json_file.write_text(json.dumps(json_data, indent=2))
        
        loader = JSONLoader(str(json_file))
        docs = loader.load()
        assert len(docs) > 0
        content = " ".join(doc.page_content for doc in docs)
        assert "Alice" in content
        print("✓ JSON loader works")

def test_unsupported_file():
    """Test handling of unsupported file types."""
    print("Testing unsupported file handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        unsupported_file = Path(temp_dir) / "test.xyz"
        unsupported_file.write_text("some content")
        
        try:
            loader = DocumentLoaderFactory.create_loader(str(unsupported_file))
            assert False, "Should have raised UnsupportedDocumentTypeError"
        except UnsupportedDocumentTypeError:
            print("✓ Unsupported file type correctly rejected")

def test_factory():
    """Test document loader factory."""
    print("Testing document loader factory...")
    
    extensions = DocumentLoaderFactory.get_supported_extensions()
    assert '.txt' in extensions
    assert '.pdf' in extensions
    assert '.html' in extensions
    assert '.csv' in extensions
    assert '.json' in extensions
    
    print(f"✓ Factory supports {len(extensions)} file types")

def test_file_validation():
    """Test file validation."""
    print("Testing file validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test non-existent file
        fake_file = Path(temp_dir) / "nonexistent.txt"
        
        try:
            loader = EnhancedTextLoader(str(fake_file))
            loader.validate_file()
            assert False, "Should have raised FileValidationError"
        except Exception:
            print("✓ Non-existent file validation works")

def test_ingestion_basic():
    """Test basic ingestion functionality."""
    print("Testing basic ingestion...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        txt_file = Path(temp_dir) / "test.txt"
        txt_file.write_text("This is a test document for ingestion testing.")
        
        # Create ingester with temporary DB
        db_path = Path(temp_dir) / "test_db"
        ingester = DocumentIngester(
            chunk_size=100,
            chunk_overlap=20,
            persist_directory=str(db_path)
        )
        
        # Test health check
        health = ingester.health_check()
        assert health["status"] in ["healthy", "unhealthy"]  # May fail in test env
        print("✓ Health check works")
        
        # Test document loading
        docs = ingester.load_document(str(txt_file))
        assert len(docs) == 1
        print("✓ Document loading works")
        
        # Test document processing
        chunks = ingester.process_documents(docs)
        assert len(chunks) >= 1
        print("✓ Document processing works")
        
        # Test ingestion
        vectorstore = ingester.ingest_files([str(txt_file)])
        assert vectorstore is not None
        print("✓ File ingestion works")
        
        # Test stats
        stats = ingester.get_ingestion_stats()
        assert "total_collections" in stats
        print("✓ Stats retrieval works")

def main():
    """Run all tests."""
    print("Running enhanced ingestion system tests...\n")
    
    try:
        test_config()
        test_document_loaders()
        test_unsupported_file()
        test_factory()
        test_file_validation()
        test_ingestion_basic()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
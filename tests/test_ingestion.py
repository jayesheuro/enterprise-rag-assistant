"""Tests for document ingestion."""

from pathlib import Path
import pytest

from app.core.exceptions import UnsupportedFileTypeError, EmbeddingModelMismatchError
from app.services.ingestion.loaders import load_text, load_markdown, load_document, LOADER_REGISTRY
from app.services.ingestion.chunkers import RecursiveCharacterChunker, MarkdownAwareChunker
from app.services.ingestion.loaders import Document
from app.services.retrieval.vector_store import VectorStore
from tests.fakes import FakeEmbedder


def test_load_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    content = "Hello world! This is a test text file."
    file_path.write_text(content, encoding="utf-8")
    
    doc = load_text(file_path)
    
    assert doc.text == content
    assert doc.metadata["filename"] == "test.txt"
    assert doc.metadata["filetype"] == "txt"
    assert doc.doc_id is not None


def test_load_markdown_file(tmp_path):
    file_path = tmp_path / "test.md"
    content = "# Heading\nMarkdown content"
    file_path.write_text(content, encoding="utf-8")
    
    doc = load_markdown(file_path)
    
    assert doc.text == content
    assert doc.metadata["filetype"] == "md"


def test_unsupported_file_type(tmp_path):
    file_path = tmp_path / "test.xyz"
    file_path.write_text("content", encoding="utf-8")
    
    with pytest.raises(UnsupportedFileTypeError):
        load_document(file_path)


def test_recursive_chunker_basic():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    doc = Document(doc_id="test1", text="A" * 100, metadata={})
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_recursive_chunker_small_doc():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    doc = Document(doc_id="test2", text="Small text", metadata={})
    chunks = chunker.chunk(doc)
    
    assert len(chunks) == 1
    assert chunks[0].text == "Small text"


def test_recursive_chunker_overlap():
    chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=5)
    doc = Document(doc_id="test3", text="1234567890 1234567890 1234567890", metadata={})
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1


def test_markdown_chunker_headings():
    chunker = MarkdownAwareChunker(chunk_size=100, chunk_overlap=10)
    content = "# Main\nContent\n## Sub\nMore content"
    doc = Document(doc_id="test4", text=content, metadata={})
    chunks = chunker.chunk(doc)
    
    assert any("Main > Sub:" in c.text for c in chunks)


def test_markdown_chunker_fallback():
    chunker = MarkdownAwareChunker(chunk_size=20, chunk_overlap=5)
    content = "# Heading\n" + "A" * 50
    doc = Document(doc_id="test5", text=content, metadata={})
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1
    assert all(len(c.text) <= 20 for c in chunks)


def test_vector_store_idempotent(tmp_path):
    db_path = tmp_path / "lancedb"
    store = VectorStore(db_path)
    
    embedder = FakeEmbedder()
    doc = Document(doc_id="doc1", text="Test content", metadata={})
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(doc)
    vectors = embedder.embed_documents([c.text for c in chunks])
    
    store.add_chunks(chunks, vectors, "fake_model")
    stats1 = store.get_stats()
    
    # Ingest again
    store.add_chunks(chunks, vectors, "fake_model")
    stats2 = store.get_stats()
    
    assert stats1["row_count"] == stats2["row_count"]
    assert stats1["doc_count"] == 1


def test_embedding_model_mismatch(tmp_path):
    db_path = tmp_path / "lancedb"
    store = VectorStore(db_path)
    
    embedder = FakeEmbedder()
    doc1 = Document(doc_id="doc1", text="Test 1", metadata={})
    doc2 = Document(doc_id="doc2", text="Test 2", metadata={})
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    
    chunks1 = chunker.chunk(doc1)
    vectors1 = embedder.embed_documents([c.text for c in chunks1])
    store.add_chunks(chunks1, vectors1, "model_a")
    
    chunks2 = chunker.chunk(doc2)
    vectors2 = embedder.embed_documents([c.text for c in chunks2])
    
    with pytest.raises(EmbeddingModelMismatchError):
        store.add_chunks(chunks2, vectors2, "model_b")


def test_loader_registry():
    assert ".txt" in LOADER_REGISTRY
    assert ".md" in LOADER_REGISTRY
    assert ".pdf" in LOADER_REGISTRY

"""Document loaders for various file types."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pypdf
from pydantic import BaseModel, Field

from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)


class Document(BaseModel):
    """Represents a loaded document before chunking."""

    doc_id: str = Field(..., description="Stable hash of path+content")
    text: str = Field(..., description="Full text content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


def _generate_doc_id(path: Path, content: str) -> str:
    """Generate a stable 16-character hash for a document."""
    raw = str(path.name) + content
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_text(path: Path) -> Document:
    """Load a plain text file."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    
    doc_id = _generate_doc_id(path, content)
    metadata = {
        "source_path": str(path),
        "filename": path.name,
        "filetype": path.suffix.lower().lstrip("."),
        "page_numbers": [],
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }
    return Document(doc_id=doc_id, text=content, metadata=metadata)


def load_markdown(path: Path) -> Document:
    """Load a markdown file."""
    return load_text(path)


def load_pdf(path: Path) -> Document:
    """Load a PDF file and extract text."""
    texts = []
    page_numbers = []
    with path.open("rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                page_numbers.append(i + 1)
                
    content = "\n".join(texts)
    doc_id = _generate_doc_id(path, content)
    metadata = {
        "source_path": str(path),
        "filename": path.name,
        "filetype": "pdf",
        "page_numbers": page_numbers,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }
    return Document(doc_id=doc_id, text=content, metadata=metadata)


LOADER_REGISTRY: dict[str, Callable[[Path], Document]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: Path) -> Document:
    """Load a document based on its extension."""
    ext = path.suffix.lower()
    loader = LOADER_REGISTRY.get(ext)
    if not loader:
        raise UnsupportedFileTypeError(ext)
    return loader(path)


def load_directory(dir_path: Path) -> list[Document]:
    """Load all supported documents in a directory."""
    documents = []
    for path in dir_path.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            if ext in LOADER_REGISTRY:
                try:
                    documents.append(load_document(path))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
            else:
                logger.debug(f"Skipping unsupported file: {path}")
    return documents

"""Document chunking strategies."""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import ChunkingConfig
from app.services.ingestion.loaders import Document


class Chunk(BaseModel):
    """A segment of a document ready for embedding."""

    chunk_id: str = Field(..., description="Hash of doc_id + chunk_index")
    doc_id: str = Field(..., description="ID of the source document")
    text: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position of chunk in document")
    token_count: int = Field(..., description="Estimated number of tokens")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata from source document")


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        pass
        
    def _create_chunk(self, document: Document, text: str, index: int) -> Chunk:
        """Helper to create a Chunk object."""
        chunk_id_raw = f"{document.doc_id}_{index}"
        chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()[:16]
        
        metadata = document.metadata.copy()
        # Basic positional info (could be enhanced)
        metadata.update({
            "page": None,  # Advanced tracking could infer page from position
            "start_pos": 0,
            "end_pos": 0
        })
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            text=text,
            chunk_index=index,
            token_count=len(text) // 4,
            metadata=metadata
        )


class RecursiveCharacterChunker(BaseChunker):
    """Chunks text recursively on paragraph, sentence, and word boundaries."""

    def chunk(self, document: Document) -> list[Chunk]:
        return self._split_text(document, document.text, 0)
        
    def _split_text(self, document: Document, text: str, start_idx: int) -> list[Chunk]:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        
        # If it already fits, return it
        if len(text) <= self.chunk_size:
            return [self._create_chunk(document, text, start_idx)]
            
        # Find the best separator
        for sep in separators:
            if sep == "":
                # Fallback to character splitting
                splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
                return [self._create_chunk(document, s, start_idx + i) for i, s in enumerate(splits)]
                
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current_text = ""
                idx = start_idx
                
                for part in parts:
                    if not current_text:
                        current_text = part
                    elif len(current_text) + len(sep) + len(part) <= self.chunk_size:
                        current_text += sep + part
                    else:
                        # Add current chunk
                        chunks.append(self._create_chunk(document, current_text.strip(), idx))
                        idx += 1
                        # Start new chunk with overlap
                        # Simple overlap approach: keep last N characters
                        overlap_start = max(0, len(current_text) - self.chunk_overlap)
                        current_text = current_text[overlap_start:] + sep + part if overlap_start > 0 else part
                        
                if current_text:
                    chunks.append(self._create_chunk(document, current_text.strip(), idx))
                    
                # Verify chunks are valid size; if not, recursive call (simplified here)
                final_chunks = []
                for i, c in enumerate(chunks):
                    if len(c.text) > self.chunk_size:
                        sub_chunks = self._split_text(document, c.text, i * 1000) # Ensure unique indices
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(c)
                
                # Re-index
                for i, c in enumerate(final_chunks):
                    c.chunk_index = start_idx + i
                    c.chunk_id = hashlib.sha256(f"{document.doc_id}_{start_idx + i}".encode("utf-8")).hexdigest()[:16]
                    
                return final_chunks
                
        return [self._create_chunk(document, text, start_idx)]


class MarkdownAwareChunker(BaseChunker):
    """Chunks Markdown text keeping heading breadcrumbs."""

    def chunk(self, document: Document) -> list[Chunk]:
        lines = document.text.split("\n")
        sections = []
        current_path = []
        current_content = []
        
        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            if match:
                # Save previous section if it has content
                if current_content:
                    sections.append((" > ".join(current_path), "\n".join(current_content)))
                    current_content = []
                
                level = len(match.group(1))
                heading = match.group(2).strip()
                
                # Update breadcrumb path
                current_path = current_path[:level - 1]
                current_path.append(heading)
            else:
                current_content.append(line)
                
        if current_content:
            sections.append((" > ".join(current_path), "\n".join(current_content)))
            
        chunks = []
        idx = 0
        recursive_chunker = RecursiveCharacterChunker(self.chunk_size, self.chunk_overlap)
        
        for breadcrumb, content in sections:
            full_text = f"{breadcrumb}: {content}" if breadcrumb else content
            
            if len(full_text) <= self.chunk_size:
                chunks.append(self._create_chunk(document, full_text, idx))
                idx += 1
            else:
                # Fallback to recursive chunker for this section
                doc_copy = Document(doc_id=document.doc_id, text=full_text, metadata=document.metadata)
                sub_chunks = recursive_chunker.chunk(doc_copy)
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.chunk_id = hashlib.sha256(f"{document.doc_id}_{idx}".encode("utf-8")).hexdigest()[:16]
                    chunks.append(sc)
                    idx += 1
                    
        if not chunks:
            # Empty document
            return []
            
        return chunks


def get_chunker(config: ChunkingConfig) -> BaseChunker:
    """Factory to get the configured chunker."""
    if config.strategy == "markdown_aware":
        return MarkdownAwareChunker(config.chunk_size, config.chunk_overlap)
    return RecursiveCharacterChunker(config.chunk_size, config.chunk_overlap)

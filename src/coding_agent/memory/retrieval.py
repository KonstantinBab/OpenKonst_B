"""Memory-aware retrieval packing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from coding_agent.memory.store import MemoryStore
from coding_agent.tools.search_tools import FileChunk, SearchTools


class RetrievalBundle(BaseModel):
    memory_hits: list[str] = Field(default_factory=list)
    file_chunks: list[FileChunk] = Field(default_factory=list)


class RetrievalService:
    """Combines memory search with lexical project retrieval."""

    def __init__(self, memory_store: MemoryStore, search_tools: SearchTools):
        self.memory_store = memory_store
        self.search_tools = search_tools

    def retrieve(self, query: str, max_chunks: int = 8) -> RetrievalBundle:
        memories = [item.content for item in self.memory_store.search_memory(query, limit=5)]
        chunks = self.search_tools.pack_context(query, max_chunks=max_chunks)
        if not chunks:
            chunks = self._default_project_context(max_chunks)
        return RetrievalBundle(memory_hits=memories, file_chunks=chunks)

    def _default_project_context(self, max_chunks: int) -> list[FileChunk]:
        chunks: list[FileChunk] = []
        for path in ("README.md", "pyproject.toml", "config/default.yaml"):
            try:
                chunks.extend(self.search_tools.chunk_file(path)[:1])
            except OSError:
                continue
            if len(chunks) >= max_chunks:
                break
        return chunks[:max_chunks]

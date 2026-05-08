"""Tool adapters for the memory store."""

from __future__ import annotations

from coding_agent.memory.models import MemoryRecord
from coding_agent.memory.store import MemoryStore


class MemoryTools:
    """Thin tool surface over the persistent memory store."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def add(self, kind: str, content: str, tags: list[str] | None = None) -> MemoryRecord:
        return self.store.add_memory(MemoryRecord(kind=kind, content=content, tags=tags or []))

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        return self.store.search_memory(query, limit=limit)


"""Retrieval-oriented file search helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.sandbox.workspace_guard import WorkspaceGuard


class SearchHit(BaseModel):
    path: str
    line_number: int
    line: str


class FileChunk(BaseModel):
    path: str
    start_line: int
    end_line: int
    content: str


class SearchTools:
    """Lexical search and chunk extraction."""

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard

    def search(self, query: str, glob: str = "*") -> list[SearchHit]:
        results: list[SearchHit] = []
        for file_path in self.guard.glob(".", glob):
            try:
                for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if query.lower() in line.lower():
                        results.append(SearchHit(path=str(file_path.relative_to(self.guard.root)), line_number=line_number, line=line.strip()))
            except UnicodeDecodeError:
                continue
        return results

    def chunk_file(self, path: str, chunk_size: int = 40, overlap: int = 5) -> list[FileChunk]:
        target = self.guard.resolve_path(path).resolved
        lines = target.read_text(encoding="utf-8").splitlines()
        chunks: list[FileChunk] = []
        start = 0
        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            content = "\n".join(lines[start:end])
            chunks.append(FileChunk(path=path, start_line=start + 1, end_line=end, content=content))
            if end == len(lines):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def pack_context(self, query: str, max_chunks: int = 8, glob: str = "*") -> list[FileChunk]:
        hits = self.search(query, glob=glob)
        packed: list[FileChunk] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.path in seen:
                continue
            packed.extend(self.chunk_file(hit.path)[:1])
            seen.add(hit.path)
            if len(packed) >= max_chunks:
                break
        return packed[:max_chunks]

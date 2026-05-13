"""Retrieval-oriented file search helpers."""

from __future__ import annotations

from pathlib import Path
import re
import time

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

    DEFAULT_SKIP_DIRS = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "artifacts",
        "data",
        "models",
        "checkpoints",
    }
    DEFAULT_TEXT_SUFFIXES = {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
        ".cfg",
        ".sh",
        ".ps1",
        ".bat",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
    }
    MAX_SEARCH_FILE_BYTES = 512_000
    MAX_SEARCH_FILES = 1500
    MAX_SEARCH_SECONDS = 8.0

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard

    def search(self, query: str, glob: str = "*") -> list[SearchHit]:
        results: list[SearchHit] = []
        needle = query.lower().strip()
        if not needle:
            return results
        started = time.monotonic()
        scanned = 0
        for file_path in self.guard.glob(".", glob):
            if not self._is_searchable_text_file(file_path):
                continue
            scanned += 1
            if scanned > self.MAX_SEARCH_FILES or time.monotonic() - started > self.MAX_SEARCH_SECONDS:
                break
            try:
                for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if needle in line.lower():
                        results.append(SearchHit(path=str(file_path.relative_to(self.guard.root)), line_number=line_number, line=line.strip()))
            except (OSError, UnicodeDecodeError):
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
        hits: list[SearchHit] = []
        for keyword in self._query_keywords(query):
            hits.extend(self.search(keyword, glob=glob))
            if len({hit.path for hit in hits}) >= max_chunks:
                break
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

    def _is_searchable_text_file(self, path: Path) -> bool:
        try:
            relative_parts = path.relative_to(self.guard.root).parts
        except ValueError:
            return False
        if any(part in self.DEFAULT_SKIP_DIRS for part in relative_parts):
            return False
        if path.suffix.lower() not in self.DEFAULT_TEXT_SUFFIXES:
            return False
        try:
            return path.stat().st_size <= self.MAX_SEARCH_FILE_BYTES
        except OSError:
            return False

    @staticmethod
    def _query_keywords(query: str) -> list[str]:
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[А-Яа-яЁё][А-Яа-яЁё0-9_]{3,}", query)
        stopwords = {
            "workspace",
            "history",
            "session",
            "задача",
            "пользователь",
            "запрос",
            "проект",
            "сессии",
            "история",
            "нужно",
            "надо",
            "сначала",
            "потом",
            "текущая",
            "режим",
            "работы",
        }
        weighted: list[str] = []
        priority = (
            "train",
            "training",
            "hyperparam",
            "learning_rate",
            "batch",
            "epoch",
            "n_steps",
            "n_epochs",
            "гиперпарамет",
            "обуч",
            "тест",
        )
        for item in candidates:
            lowered = item.lower()
            if lowered in stopwords or len(lowered) > 40:
                continue
            weighted.append(lowered)
        weighted.sort(key=lambda value: 0 if any(token in value for token in priority) else 1)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in weighted:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:8] or [query[:80]]

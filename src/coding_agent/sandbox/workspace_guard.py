"""Workspace boundary enforcement."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.util.errors import WorkspaceViolationError


class GuardedPath(BaseModel):
    raw: str
    resolved: Path
    relative: str


class WorkspaceGuard:
    """Resolves paths and guarantees they stay inside the configured workspace."""

    def __init__(self, root: Path | str, deny_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.deny_patterns = deny_patterns or []
        self.ignore_patterns = ignore_patterns or []
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str | Path) -> GuardedPath:
        candidate = Path(path)
        resolved = (self.root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolationError(f"Path escapes workspace: {path}") from exc
        relative = resolved.relative_to(self.root).as_posix()
        self._check_patterns(relative)
        return GuardedPath(raw=str(path), resolved=resolved, relative=relative)

    def ensure_cwd(self, cwd: str | Path | None) -> Path:
        if cwd is None:
            return self.root
        return self.resolve_path(cwd).resolved

    def is_ignored(self, path: str | Path) -> bool:
        relative = self.resolve_path(path).relative
        return any(fnmatch(relative, pattern) for pattern in self.ignore_patterns)

    def glob(self, base: str | Path = ".", pattern: str = "*") -> list[Path]:
        base_path = self.resolve_path(base).resolved
        results: list[Path] = []
        for path in base_path.rglob(pattern):
            relative = path.relative_to(self.root).as_posix()
            if path.is_file() and not self._matches_any(relative, self.ignore_patterns + self.deny_patterns):
                results.append(path)
        return results

    def _check_patterns(self, relative: str) -> None:
        if self._matches_any(relative, self.deny_patterns):
            raise WorkspaceViolationError(f"Path denied by workspace policy: {relative}")

    @staticmethod
    def _matches_any(value: str, patterns: list[str]) -> bool:
        return any(fnmatch(value, pattern) for pattern in patterns)


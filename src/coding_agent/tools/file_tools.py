"""Workspace-safe file operations."""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.util.errors import ToolExecutionError


class AuditEntry(BaseModel):
    action: str
    path: str
    timestamp: str


class FileToolResult(BaseModel):
    ok: bool
    action: str
    path: str
    message: str = ""
    content: str | None = None
    paths: list[str] = Field(default_factory=list)
    diff_preview: str | None = None
    audit: AuditEntry | None = None


class FileTools:
    """Safe file tools built on workspace guard and policy enforcement."""

    def __init__(self, guard: WorkspaceGuard, policy: CommandPolicyEngine):
        self.guard = guard
        self.policy = policy

    def list_files(self, path: str = ".", glob: str = "*") -> FileToolResult:
        self._enforce("list")
        files = [str(item.relative_to(self.guard.root)) for item in self.guard.glob(path, glob)]
        return self._result("list_files", path, paths=sorted(files))

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None, max_chars: int = 12000) -> FileToolResult:
        self._enforce("read")
        target = self.guard.resolve_path(path).resolved
        lines = target.read_text(encoding="utf-8").splitlines()
        start_index = max((start_line or 1) - 1, 0)
        end_index = min(end_line or len(lines), len(lines))
        if end_index < start_index:
            raise ToolExecutionError("end_line must be greater than or equal to start_line")
        snippet = "\n".join(lines[start_index:end_index])[:max_chars]
        return self._result("read_file", path, content=snippet)

    def write_file(self, path: str, content: str, approval: bool = False) -> FileToolResult:
        self._enforce("write", approval)
        target = self.guard.resolve_path(path).resolved
        target.parent.mkdir(parents=True, exist_ok=True)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        if self._looks_like_accidental_truncation(target, old, content, approval):
            raise ToolExecutionError(
                f"Refusing to overwrite existing source file {path} with much smaller content. "
                "Use replace_in_file or apply_unified_diff for targeted edits."
            )
        target.write_text(content, encoding="utf-8")
        preview = self._preview(old, content)
        return self._result("write_file", path, diff_preview=preview)

    def replace_in_file(self, path: str, old: str, new: str, approval: bool = False) -> FileToolResult:
        self._enforce("write", approval)
        target = self.guard.resolve_path(path).resolved
        current = target.read_text(encoding="utf-8")
        if old not in current:
            raise ToolExecutionError(f"Target text was not found in {path}")
        updated = current.replace(old, new)
        target.write_text(updated, encoding="utf-8")
        return self._result("replace_in_file", path, diff_preview=self._preview(current, updated))

    def move_file(self, src: str, dst: str, approval: bool = False) -> FileToolResult:
        self._enforce("move", approval)
        source = self.guard.resolve_path(src).resolved
        target = self.guard.resolve_path(dst).resolved
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        return self._result("move_file", src, message=f"Moved to {target.relative_to(self.guard.root)}")

    def delete_file(self, path: str, approval: bool = False) -> FileToolResult:
        self._enforce("delete", approval)
        target = self.guard.resolve_path(path).resolved
        if target.exists():
            target.unlink()
        return self._result("delete_file", path)

    def mkdir(self, path: str, approval: bool = False) -> FileToolResult:
        self._enforce("mkdir", approval)
        target = self.guard.resolve_path(path).resolved
        target.mkdir(parents=True, exist_ok=True)
        return self._result("mkdir", path)

    def search_in_files(self, query: str, glob: str = "*") -> FileToolResult:
        self._enforce("search")
        matches: list[str] = []
        for file_path in self.guard.glob(".", glob):
            try:
                for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if query.lower() in line.lower():
                        matches.append(f"{file_path.relative_to(self.guard.root)}:{line_number}:{line.strip()}")
            except UnicodeDecodeError:
                continue
        return self._result("search_in_files", ".", paths=matches)

    def _enforce(self, action: str, approval: bool = False) -> None:
        decision = self.policy.evaluate_file_op(action)
        self.policy.enforce(decision, approval=approval)

    def _result(self, action: str, path: str, **kwargs: object) -> FileToolResult:
        return FileToolResult(
            ok=True,
            action=action,
            path=path,
            audit=AuditEntry(action=action, path=path, timestamp=datetime.now(UTC).isoformat()),
            **kwargs,
        )

    @staticmethod
    def _preview(old: str, new: str, limit: int = 2000) -> str:
        if old == new:
            return "No changes."
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        return diff[:limit]

    @staticmethod
    def _looks_like_accidental_truncation(target: Path, old: str, new: str, approval: bool) -> bool:
        if approval or not old:
            return False
        if target.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml", ".toml"}:
            return False
        old_len = len(old.strip())
        new_len = len(new.strip())
        if old_len < 1000:
            return False
        return new_len < max(300, old_len // 4)

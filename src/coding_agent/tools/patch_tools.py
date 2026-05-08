"""Patch-first file modification tools."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from pydantic import BaseModel

from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.tools.file_tools import FileTools, FileToolResult
from coding_agent.util.errors import ToolExecutionError


class PatchApplyResult(BaseModel):
    ok: bool
    path: str
    preview: str
    message: str


class PatchTools:
    """Applies unified diffs or replace-based fallbacks within the workspace."""

    def __init__(self, guard: WorkspaceGuard, file_tools: FileTools):
        self.guard = guard
        self.file_tools = file_tools

    def apply_unified_diff(self, path: str, diff: str, approval: bool = False) -> PatchApplyResult:
        target = self.guard.resolve_path(path).resolved
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        patched = self._apply_simple_unified_diff(original, diff)
        preview = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                patched.splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        self.file_tools.write_file(path, patched, approval=approval)
        return PatchApplyResult(ok=True, path=path, preview=preview[:4000], message="Patch applied")

    def replace_patch(self, path: str, old: str, new: str, approval: bool = False) -> FileToolResult:
        return self.file_tools.replace_in_file(path, old, new, approval=approval)

    @staticmethod
    def _apply_simple_unified_diff(original: str, diff_text: str) -> str:
        lines = original.splitlines()
        updated: list[str] = []
        index = 0
        hunk_active = False
        for raw_line in diff_text.splitlines():
            if raw_line.startswith(("---", "+++")):
                continue
            if raw_line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
                if not match:
                    raise ToolExecutionError("Invalid unified diff hunk header")
                source_start = int(match.group(1)) - 1
                updated.extend(lines[index:source_start])
                index = source_start
                hunk_active = True
                continue
            if not hunk_active:
                continue
            if raw_line.startswith(" "):
                if index >= len(lines) or lines[index] != raw_line[1:]:
                    raise ToolExecutionError("Patch context mismatch")
                updated.append(lines[index])
                index += 1
            elif raw_line.startswith("-"):
                if index >= len(lines) or lines[index] != raw_line[1:]:
                    raise ToolExecutionError("Patch removal mismatch")
                index += 1
            elif raw_line.startswith("+"):
                updated.append(raw_line[1:])
        updated.extend(lines[index:])
        return "\n".join(updated) + ("\n" if original.endswith("\n") or diff_text.endswith("\n") else "")

from pathlib import Path

import pytest

from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.util.errors import WorkspaceViolationError


def test_workspace_guard_blocks_escape(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceViolationError):
        guard.resolve_path("..\\outside.txt")


def test_workspace_guard_resolves_inside_workspace(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guarded = guard.resolve_path("folder\\file.txt")
    assert guarded.resolved == tmp_path.joinpath("folder", "file.txt").resolve()


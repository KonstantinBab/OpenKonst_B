"""Run and chat session models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    run_id: str
    goal: str
    step_count: int = 0
    tool_calls: int = 0
    events: list[str] = Field(default_factory=list)
    last_test_passed: bool | None = None
    last_test_output: str = ""


class ChatHistoryEntry(BaseModel):
    role: str
    content: str


class ChatSessionState(BaseModel):
    current_goal: str = ""
    active_task: str = ""
    task_status: str = "idle"
    last_summary: str = ""
    last_changed_files: list[str] = Field(default_factory=list)
    pending_verifications: list[str] = Field(default_factory=list)
    messages: list[ChatHistoryEntry] = Field(default_factory=list)
    turn_count: int = 0


class ChatRunResult(BaseModel):
    status: str
    summary: str
    changes: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    findings: list[dict[str, str]] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    assistant_message: str = ""

"""Run session models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    run_id: str
    goal: str
    step_count: int = 0
    tool_calls: int = 0
    events: list[str] = Field(default_factory=list)


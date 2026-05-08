"""Pydantic models for persisted memory."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MemoryKind = Literal["episodic", "semantic", "procedural", "working"]


class MemoryRecord(BaseModel):
    id: int | None = None
    kind: MemoryKind
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class RunRecord(BaseModel):
    id: int | None = None
    run_id: str
    goal: str
    status: str
    summary: str = ""
    created_at: datetime | None = None


class ToolCallRecord(BaseModel):
    id: int | None = None
    run_id: str
    tool_name: str
    input_json: str
    output_json: str
    created_at: datetime | None = None


class CommitRecord(BaseModel):
    id: int | None = None
    run_id: str
    commit_hash: str
    message: str
    created_at: datetime | None = None


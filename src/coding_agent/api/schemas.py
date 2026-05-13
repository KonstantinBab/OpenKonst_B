"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class MemorySearchRequest(BaseModel):
    workspace: str
    query: str
    limit: int = 10


class ChatRequest(BaseModel):
    workspace: str = "."
    message: str
    model: str | None = None
    russian_only: bool = True
    timeout_seconds: int = 180


class ChatResponse(BaseModel):
    status: str
    assistant_message: str
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    report_path: str
    state_path: str


class RunListResponse(BaseModel):
    runs: list[dict] = Field(default_factory=list)


class RepoScanResponse(BaseModel):
    artifact_path: str
    repo_scan: dict


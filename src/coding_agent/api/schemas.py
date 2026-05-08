"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class MemorySearchRequest(BaseModel):
    workspace: str
    query: str
    limit: int = 10


class RunListResponse(BaseModel):
    runs: list[dict] = Field(default_factory=list)


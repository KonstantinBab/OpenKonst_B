"""FastAPI scaffold."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from coding_agent.api.schemas import HealthResponse, MemorySearchRequest, RunListResponse
from coding_agent.core.bootstrap import build_runtime
from coding_agent.util.logging import configure_logging

configure_logging()
app = FastAPI(title="coding-agent")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/runs", response_model=RunListResponse)
def runs(workspace: str = ".") -> RunListResponse:
    runtime = build_runtime(Path(workspace))
    return RunListResponse(runs=[record.model_dump(mode="json") for record in runtime.memory_store.list_runs()])


@app.post("/memory/search")
def memory_search(request: MemorySearchRequest) -> list[dict]:
    runtime = build_runtime(Path(request.workspace))
    return [record.model_dump(mode="json") for record in runtime.memory_tools.search(request.query, request.limit)]


"""Project context aggregation."""

from __future__ import annotations

from pydantic import BaseModel

from coding_agent.memory.retrieval import RetrievalBundle


class ProjectContext(BaseModel):
    workspace: str
    retrieval_bundle: RetrievalBundle
    project_overview: str = ""
    repo_map: str = ""
    recent_observations: list[str] = []

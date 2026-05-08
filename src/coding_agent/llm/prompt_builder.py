"""Prompt construction for the orchestrator."""

from __future__ import annotations

from coding_agent.core.context import ProjectContext
from coding_agent.memory.summarizer import summarize_retrieval


def build_turn_prompt(goal: str, context: ProjectContext) -> str:
    """Build a compact structured-action prompt."""
    retrieval_text = summarize_retrieval(context.retrieval_bundle)
    observations_text = "\n".join(f"- {item}" for item in context.recent_observations[-8:])
    return (
        "You are a coding agent. Return valid JSON only.\n"
        "Do not use markdown fences. Do not add commentary outside JSON.\n"
        "Decide the next tool actions needed to progress toward the goal.\n"
        f"Goal:\n{goal}\n\n"
        f"Workspace:\n{context.workspace}\n\n"
        f"Retrieved context:\n{retrieval_text}\n\n"
        f"Recent tool observations:\n{observations_text or 'None yet.'}\n\n"
        "Allowed actions: list_files, read_file, search_in_files, write_file, replace_in_file, move_file, "
        "delete_file, mkdir, apply_unified_diff, run_command, git_status, git_diff, git_create_branch, "
        "git_add, git_commit, memory_search, memory_add.\n"
        "Response schema:\n"
        '{"plan":[{"title":"string","status":"pending|in_progress|completed"}],'
        '"actions":[{"action":"allowed_action","arguments":{}}],'
        '"final_report":{"status":"completed|stopped|failed","summary":"string","changes":["string"],"verification":["string"]}}\n'
        "Use either actions for the next step or final_report when the task is complete."
    )

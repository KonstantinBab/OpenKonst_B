"""Prompt construction for the orchestrator."""

from __future__ import annotations

from coding_agent.core.context import ProjectContext
from coding_agent.memory.summarizer import summarize_retrieval


def build_turn_prompt(goal: str, context: ProjectContext) -> str:
    """Build a compact structured-action prompt."""
    retrieval_text = summarize_retrieval(context.retrieval_bundle)[:8000]
    project_overview = (context.project_overview or "No overview yet.")[:5000]
    repo_map = (context.repo_map or "No repo map yet.")[:5000]
    observations_text = "\n".join(f"- {item}" for item in context.recent_observations[-4:])[:4000]
    lowered_goal = goal.lower()
    action_goal = any(
        keyword in lowered_goal
        for keyword in (
            "измени",
            "изменить",
            "исправ",
            "почини",
            "добав",
            "реализ",
            "сделай",
            "обнов",
            "перепиш",
            "отрефактор",
            "запусти",
            "запустить",
            "обучи",
            "обучение",
            "установ",
            "install",
            "fix",
            "change",
            "update",
            "implement",
            "run",
            "train",
        )
    )
    analytical_goal = not action_goal and any(keyword in lowered_goal for keyword in ("риск", "risk", "проверк", "audit", "analy", "анализ"))
    return (
        "You are a coding agent. Return valid JSON only.\n"
        "Do not use markdown fences. Do not add commentary outside JSON.\n"
        "Use Russian language for all human-readable fields (plan titles, summary, changes, verification).\n"
        "Decide the next tool actions needed to progress toward the goal.\n"
        + (
            "This is an analytical task. Prefer evidence gathering before final_report. "
            "Do not claim facts without evidence from read_file, search_in_files, git_diff, git_status, or run_command. "
            "When you finish, findings must be concrete and evidence-backed.\n"
            if analytical_goal
            else ""
        )
        + (
            "This is an implementation/execution task. Do not answer with advice only. "
            "First inspect the relevant files with read_file/search_in_files/list_files, then use write_file, "
            "replace_in_file, apply_unified_diff, or run_command as needed. "
            "Only produce final_report after you have either changed code, run the requested command, "
            "or collected a concrete blocker/error from a tool.\n"
            if action_goal
            else ""
        )
        + (
            "The user explicitly requested command execution. You must use run_command before final_report, "
            "unless a prior tool observation gives a concrete blocker that prevents command execution.\n"
            if any(keyword in lowered_goal for keyword in ("запусти", "запустить", "обучи", "обучение", "run", "train", "install", "установ"))
            else ""
        )
        + 
        f"Goal:\n{goal}\n\n"
        f"Workspace:\n{context.workspace}\n\n"
        f"Project overview:\n{project_overview}\n\n"
        f"Repository map:\n{repo_map}\n\n"
        f"Retrieved context:\n{retrieval_text}\n\n"
        f"Recent tool observations:\n{observations_text or 'None yet.'}\n\n"
        "Allowed actions: list_files, read_file, search_in_files, write_file, replace_in_file, move_file, "
        "delete_file, mkdir, apply_unified_diff, run_command, git_status, git_diff, git_create_branch, "
        "git_add, git_commit, memory_search, memory_add.\n"
        "Response schema:\n"
        '{"plan":[{"title":"string","status":"pending|in_progress|completed"}],'
        '"actions":[{"action":"allowed_action","arguments":{}}],'
        '"final_report":{"status":"completed|stopped|failed","summary":"string","changes":["string"],"verification":["string"],'
        '"findings":[{"title":"string","severity":"low|medium|high","status":"observed|verified|hypothesis","evidence":"string","recommendation":"string"}],'
        '"next_steps":["string"]}}\n'
        "Use either actions for the next step or final_report when the task is complete."
    )

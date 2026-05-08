"""Agent orchestrator."""

from __future__ import annotations

import json
import uuid
from typing import Any

from coding_agent.config.settings import AppConfig
from coding_agent.core.context import ProjectContext
from coding_agent.core.planner import AgentTurn, FinalReport
from coding_agent.core.session import SessionState
from coding_agent.core.verifier import Verifier
from coding_agent.llm.base import BaseLLMProvider, ChatMessage
from coding_agent.llm.prompt_builder import build_turn_prompt
from coding_agent.memory.models import RunRecord, ToolCallRecord
from coding_agent.memory.retrieval import RetrievalService
from coding_agent.memory.store import MemoryStore
from coding_agent.tools.file_tools import FileTools
from coding_agent.tools.git_tools import GitManager
from coding_agent.tools.memory_tools import MemoryTools
from coding_agent.tools.patch_tools import PatchTools
from coding_agent.tools.search_tools import SearchTools
from coding_agent.sandbox.shell_runner import ShellCommand, ShellRunner
from coding_agent.util.errors import LLMResponseError
from coding_agent.util.json_utils import extract_json_object


class Orchestrator:
    """Bounded structured-action loop."""

    def __init__(
        self,
        config: AppConfig,
        llm: BaseLLMProvider,
        memory_store: MemoryStore,
        retrieval_service: RetrievalService,
        file_tools: FileTools,
        patch_tools: PatchTools,
        git_manager: GitManager,
        memory_tools: MemoryTools,
        search_tools: SearchTools,
        shell_runner: ShellRunner,
        verifier: Verifier,
    ):
        self.config = config
        self.llm = llm
        self.memory_store = memory_store
        self.retrieval_service = retrieval_service
        self.file_tools = file_tools
        self.patch_tools = patch_tools
        self.git_manager = git_manager
        self.memory_tools = memory_tools
        self.search_tools = search_tools
        self.shell_runner = shell_runner
        self.verifier = verifier

    def run(self, goal: str) -> FinalReport:
        state = SessionState(run_id=str(uuid.uuid4()), goal=goal)
        self.memory_store.record_run(RunRecord(run_id=state.run_id, goal=goal, status="running"))
        final_report: FinalReport | None = None

        for _ in range(self.config.orchestrator.max_steps):
            if state.tool_calls >= self.config.orchestrator.max_tool_calls:
                break
            context = ProjectContext(
                workspace=str(self.file_tools.guard.root),
                retrieval_bundle=self.retrieval_service.retrieve(goal, self.config.orchestrator.retrieval_max_chunks),
                recent_observations=state.events,
            )
            turn = self._next_turn(goal, context)
            if turn.final_report:
                final_report = turn.final_report
                break
            for action in turn.actions:
                try:
                    output = self._execute_action(action.action, action.arguments)
                except Exception as exc:  # noqa: BLE001
                    output = {"ok": False, "action": action.action, "error": str(exc)}
                    self.memory_tools.add(
                        "working",
                        f"Tool failure during run {state.run_id}: {action.action}: {exc}",
                        tags=["tool_error"],
                    )
                state.tool_calls += 1
                self.memory_store.record_tool_call(
                    ToolCallRecord(
                        run_id=state.run_id,
                        tool_name=action.action,
                        input_json=json.dumps(action.arguments),
                        output_json=json.dumps(output, default=str),
                    )
                )
                state.events.append(self._summarize_observation(action.action, output))
            state.step_count += 1

        if final_report is None:
            final_report = self._finalize_from_observations(goal, state.events)
        self.memory_tools.add(
            "episodic",
            f"Run {state.run_id} finished with status={final_report.status}: {final_report.summary}",
            tags=["run", final_report.status],
        )
        self.memory_store.record_run(
            RunRecord(run_id=state.run_id, goal=goal, status=final_report.status, summary=final_report.summary)
        )
        return final_report

    def _next_turn(self, goal: str, context: ProjectContext) -> AgentTurn:
        prompt = build_turn_prompt(goal, context)
        response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
        try:
            payload = extract_json_object(response.content)
            return AgentTurn.from_model_payload(payload)
        except Exception as exc:  # noqa: BLE001
            raise LLMResponseError(f"Failed to parse structured agent turn: {response.content}") from exc

    def _finalize_from_observations(self, goal: str, observations: list[str]) -> FinalReport:
        verification = self.verifier.verify().results
        verification_summary = [f"{item.command}: {item.exit_code}" for item in verification]
        prompt = (
            "Return valid JSON only. Do not use markdown fences.\n"
            "Create the final report for this coding-agent run from the tool observations.\n"
            f"Goal:\n{goal}\n\n"
            "Tool observations:\n"
            + "\n".join(f"- {item}" for item in observations[-12:])
            + "\n\n"
            "Response schema:\n"
            '{"status":"completed|stopped|failed","summary":"string","changes":["string"],"verification":["string"]}'
        )
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
            payload = extract_json_object(response.content)
            report = FinalReport.model_validate(payload)
            if report.summary.strip():
                report.verification.extend(verification_summary)
                return report
        except Exception:
            pass
        successful_observations = [item for item in observations if '"ok": true' in item.lower()]
        if successful_observations:
            return FinalReport(
                status="completed",
                summary="Completed by bounded loop using successful tool observations.",
                verification=verification_summary,
            )
        return FinalReport(
            status="stopped",
            summary="Stopped after bounded loop without explicit model completion.",
            verification=verification_summary,
        )

    def _execute_action(self, action: str, arguments: dict[str, Any]) -> Any:
        if action == "list_files":
            return self.file_tools.list_files(**arguments).model_dump()
        if action == "read_file":
            return self.file_tools.read_file(**arguments).model_dump()
        if action == "search_in_files":
            return self.file_tools.search_in_files(**arguments).model_dump()
        if action == "write_file":
            return self.file_tools.write_file(**arguments).model_dump()
        if action == "replace_in_file":
            return self.file_tools.replace_in_file(**arguments).model_dump()
        if action == "move_file":
            return self.file_tools.move_file(**arguments).model_dump()
        if action == "delete_file":
            return self.file_tools.delete_file(**arguments).model_dump()
        if action == "mkdir":
            return self.file_tools.mkdir(**arguments).model_dump()
        if action == "apply_unified_diff":
            return self.patch_tools.apply_unified_diff(**arguments).model_dump()
        if action == "run_command":
            return self.shell_runner.run(ShellCommand(**arguments)).model_dump()
        if action == "git_status":
            return self.git_manager.status().model_dump()
        if action == "git_diff":
            return self.git_manager.diff(**arguments).model_dump()
        if action == "git_create_branch":
            return self.git_manager.create_branch(**arguments).model_dump()
        if action == "git_add":
            return self.git_manager.add_paths(**arguments).model_dump()
        if action == "git_commit":
            result = self.git_manager.commit(**arguments)
            if result.ok and result.stdout.strip():
                self.memory_tools.add("episodic", f"Commit created: {result.stdout.strip()}", tags=["git", "commit"])
            return result.model_dump()
        if action == "memory_search":
            return [item.model_dump() for item in self.memory_tools.search(**arguments)]
        if action == "memory_add":
            return self.memory_tools.add(**arguments).model_dump()
        raise ValueError(f"Unsupported action: {action}")

    @staticmethod
    def _summarize_observation(action: str, output: Any) -> str:
        text = json.dumps(output, ensure_ascii=False, default=str)
        return f"{action}: {text[:1200]}"

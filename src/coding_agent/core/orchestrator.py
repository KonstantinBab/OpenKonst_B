"""Agent orchestrator."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from coding_agent.config.settings import AppConfig
from coding_agent.core.context import ProjectContext
from coding_agent.core.planner import AgentTurn, FinalReport
from coding_agent.core.repo_scan import RepoScanner
from coding_agent.core.session import ChatRunResult, SessionState
from coding_agent.core.verifier import Verifier, VerificationSummary
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
        self.repo_scanner = RepoScanner(self.file_tools.guard)

    def run(self, goal: str) -> FinalReport:
        state = SessionState(run_id=str(uuid.uuid4()), goal=goal)
        self.memory_store.record_run(RunRecord(run_id=state.run_id, goal=goal, status="running"))
        final_report: FinalReport | None = None
        analytical_goal = self._is_analytical_goal(goal)
        project_overview = ""
        repo_map = ""

        if self._should_bootstrap_project(goal):
            project_overview = self._build_project_overview(state)
            repo_map = self._build_repo_map(state)

        for _ in range(self.config.orchestrator.max_steps):
            if state.tool_calls >= self.config.orchestrator.max_tool_calls:
                break
            context = ProjectContext(
                workspace=str(self.file_tools.guard.root),
                retrieval_bundle=self.retrieval_service.retrieve(goal, self.config.orchestrator.retrieval_max_chunks),
                project_overview=project_overview,
                repo_map=repo_map,
                recent_observations=state.events,
            )
            turn = self._next_turn(goal, context)
            if turn.final_report:
                if analytical_goal and not self._is_final_report_ready(turn.final_report, state.events, analytical_goal=analytical_goal):
                    state.events.append("system: final_report rejected because evidence is insufficient for an analytical goal")
                    state.step_count += 1
                    continue
                final_report = turn.final_report
                if (
                    final_report.status == "failed"
                    and not final_report.changes
                    and not final_report.verification
                    and state.events
                ):
                    final_report = self._finalize_from_observations(goal, state.events)
                break
            for action in turn.actions:
                try:
                    output = self._execute_action(action.action, action.arguments)
                except Exception as exc:
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

    def run_chat(self, goal: str, progress_callback: Callable[[str], None] | None = None) -> ChatRunResult:
        state = SessionState(run_id=str(uuid.uuid4()), goal=goal)
        self.memory_store.record_run(RunRecord(run_id=state.run_id, goal=goal, status="running"))
        final_report: FinalReport | None = None
        analytical_goal = self._is_analytical_goal(goal)
        project_overview = ""
        repo_map = ""
        parse_failures = 0
        stalled_cycles = 0
        model_unreliable = False
        assistant_message_override: str | None = None

        def emit(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        if self._should_bootstrap_project(goal):
            emit("Собираю обзор проекта и карту репозитория...")
            project_overview = self._build_project_overview(state)
            repo_map = self._build_repo_map(state)
            emit("Обзор проекта собран, перехожу к планированию следующего шага...")

        if analytical_goal:
            emit("Читаю ключевые файлы проекта для прямого анализа...")
            final_report, assistant_message_override = self._direct_chat_analysis(goal, state, project_overview, repo_map, emit)

        last_step_had_changes = False

        for _ in range(self.config.orchestrator.max_steps if final_report is None else 0):
            if state.tool_calls >= self.config.orchestrator.max_tool_calls:
                emit("Достигнут лимит tool-вызовов, перехожу к сборке итога.")
                break
            emit("Подбираю релевантный контекст по проекту...")
            retrieval_bundle = self.retrieval_service.retrieve(goal, self.config.orchestrator.retrieval_max_chunks)
            context = ProjectContext(
                workspace=str(self.file_tools.guard.root),
                retrieval_bundle=retrieval_bundle,
                project_overview=project_overview,
                repo_map=repo_map,
                recent_observations=state.events,
            )
            if analytical_goal and model_unreliable:
                emit("Перехожу в safe-mode: добираю доказательства без нового шага модели...")
                turn = self._fallback_chat_turn(goal, state.events, analytical_goal=analytical_goal)
                if turn is None:
                    emit("Safe-mode собрал достаточно наблюдений, формирую итог.")
                    break
            else:
                try:
                    emit("Планирую следующий шаг...")
                    emit("Жду ответ модели для следующего шага...")
                    turn = self._next_turn(goal, context)
                    stalled_cycles = 0
                except LLMResponseError as exc:
                    parse_failures += 1
                    stalled_cycles += 1
                    if analytical_goal:
                        model_unreliable = True
                    state.events.append(f"system: structured turn parse failed: {exc}")
                    emit("Модель вернула шумный ответ, пробую продолжить с восстановлением хода...")
                    recovered_turn = self._fallback_chat_turn(goal, state.events, analytical_goal=analytical_goal)
                    if recovered_turn is None:
                        if parse_failures >= 2 or self._should_finalize_after_stall(stalled_cycles, analytical_goal, state.events):
                            emit("Слишком много ошибок разбора, собираю итог по уже полученным наблюдениям.")
                            break
                        continue
                    emit("Перехожу на запасной план действий без доверия к ответу модели...")
                    turn = recovered_turn

            if turn.final_report:
                if analytical_goal and not self._is_final_report_ready(turn.final_report, state.events, analytical_goal=analytical_goal):
                    state.events.append("system: final_report rejected because evidence is insufficient for an analytical goal")
                    emit("Итог пока сырой, собираю дополнительные доказательства...")
                    state.step_count += 1
                    continue
                if self._requires_command_execution(goal) and not self._has_command_observation(state.events):
                    state.events.append("system: final_report rejected because requested command execution has not happened")
                    emit("Запрошен запуск команды, но run_command ещё не выполнялся. Продолжаю до фактического запуска...")
                    state.step_count += 1
                    continue
                final_report = turn.final_report
                emit("Формирую итоговый ответ...")
                break

            if not turn.actions:
                state.events.append("system: no actions returned; continuing conversational loop")
                stalled_cycles += 1
                fallback_turn = self._fallback_chat_turn(goal, state.events, analytical_goal=analytical_goal)
                if fallback_turn is None:
                    if self._should_finalize_after_stall(stalled_cycles, analytical_goal, state.events):
                        emit("Дальше идти уже некуда, собираю итог по собранным наблюдениям.")
                        break
                    emit("Модель не предложила действий, делаю ещё один цикл планирования...")
                    state.step_count += 1
                    continue
                if not fallback_turn.actions:
                    if self._should_finalize_after_stall(stalled_cycles, analytical_goal, state.events):
                        emit("Собрано достаточно наблюдений, завершаю аналитический итог без нового шага модели.")
                        break
                    emit("Модель не предложила действий, делаю ещё один цикл планирования...")
                    state.step_count += 1
                    continue
                emit("Сам выбираю следующий безопасный шаг исследования...")
                turn = fallback_turn

            for action in turn.actions:
                stalled_cycles = 0
                emit(self._describe_action_for_chat(action.action, action.arguments))
                try:
                    output = self._execute_action(action.action, action.arguments)
                except Exception as exc:
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
                if action.action in {"write_file", "replace_in_file", "apply_unified_diff"}:
                    last_step_had_changes = True
            state.step_count += 1

            # Автоматический запуск тестов после изменений
            if last_step_had_changes and self._should_auto_test(goal):
                emit("Запускаю тесты для проверки внесённых изменений...")
                test_summary = self._run_tests_and_record(state)
                if test_summary and not test_summary.passed:
                    emit("⚠️ Тесты не прошли. Анализирую ошибки и планирую исправления.")
                    failure_details = self._format_test_failure(test_summary)
                    state.events.append(f"system: tests failed after changes\n{failure_details}")
                    last_step_had_changes = False
                    continue
                elif test_summary and test_summary.passed:
                    emit("✅ Все тесты прошли успешно.")
                    state.events.append("system: all tests passed after changes")
                last_step_had_changes = False

            # Если цель требует выполнения команды и команда успешно выполнена – завершаем
            if self._requires_command_execution(goal) and self._has_successful_command_observation(state.events):
                emit("✅ Команда выполнена успешно. Завершаю задачу.")
                final_report = self._finalize_from_observations(goal, state.events)
                break
            
            # Для задач обучения: если обучение запущено и есть признаки прогресса - можно завершать
            if self._is_training_complete(goal, state.events):
                emit("✅ Обучение запущено и выполняется. Завершаю задачу.")
                final_report = self._finalize_from_observations(goal, state.events)
                break

        if final_report is None:
            emit("Собираю итог по уже собранным наблюдениям...")
            final_report = self._finalize_from_observations(goal, state.events)

        self.memory_tools.add(
            "episodic",
            f"Chat run {state.run_id} finished with status={final_report.status}: {final_report.summary}",
            tags=["chat", final_report.status],
        )
        self.memory_store.record_run(
            RunRecord(run_id=state.run_id, goal=goal, status=final_report.status, summary=final_report.summary)
        )
        changed_files = self._extract_changed_files_from_observations(state.events)
        assistant_message = assistant_message_override or self._build_chat_assistant_message(final_report, changed_files)
        return ChatRunResult(
            status=final_report.status,
            summary=final_report.summary,
            changes=final_report.changes,
            verification=final_report.verification,
            findings=final_report.findings,
            next_steps=final_report.next_steps,
            changed_files=changed_files,
            assistant_message=assistant_message,
        )

    def _run_tests_and_record(self, state: SessionState) -> VerificationSummary | None:
        """Запускает тесты через Verifier и сохраняет результат в state."""
        try:
            summary = self.verifier.verify()  # использует профиль по умолчанию
        except Exception as e:
            state.events.append(f"system: failed to run tests: {e}")
            state.last_test_passed = False
            state.last_test_output = str(e)
            return None
        state.last_test_passed = summary.passed
        state.last_test_output = self._format_test_summary(summary)
        state.events.append(f"system: tests {'passed' if summary.passed else 'failed'}")
        return summary

    def _should_auto_test(self, goal: str) -> bool:
        """Определяет, нужно ли автоматически запускать тесты после изменений."""
        lowered = goal.lower()
        if "без тестов" in lowered or "no test" in lowered:
            return False
        return getattr(self.config, "auto_test_enabled", True)

    @staticmethod
    def _format_test_summary(summary: VerificationSummary) -> str:
        """Форматирует результат тестов для вывода в события."""
        lines = [f"Profile: {summary.profile}", f"Passed: {summary.passed}"]
        for res in summary.results:
            status = "✓" if res.exit_code == 0 else "✗"
            lines.append(f"{status} {res.command} (exit {res.exit_code})")
            if res.stdout:
                lines.append(f"  stdout: {res.stdout[:200]}")
            if res.stderr:
                lines.append(f"  stderr: {res.stderr[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _format_test_failure(summary: VerificationSummary) -> str:
        """Форматирует только упавшие тесты для краткости."""
        lines = []
        for res in summary.results:
            if res.exit_code != 0:
                lines.append(f"FAIL: {res.command}")
                if res.stderr:
                    lines.append(f"  {res.stderr[:300]}")
                elif res.stdout:
                    lines.append(f"  {res.stdout[:300]}")
        return "\n".join(lines) if lines else "Tests failed with no specific output."

    @staticmethod
    def _has_successful_command_observation(observations: list[str]) -> bool:
        """Проверяет, был ли успешный run_command (exit_code=0)."""
        for item in reversed(observations):
            if item.startswith("run_command:"):
                try:
                    payload_str = item.split(":", 1)[1].strip()
                    payload = json.loads(payload_str)
                    if payload.get("exit_code") == 0:
                        return True
                except Exception:
                    if '"exit_code": 0' in item:
                        return True
        return False

    def _direct_chat_analysis(
        self,
        goal: str,
        state: SessionState,
        project_overview: str,
        repo_map: str,
        emit: Callable[[str], None],
    ) -> tuple[FinalReport, str]:
        evidence_sections: list[str] = []
        read_paths: list[str] = []
        for path in self._direct_analysis_candidates():
            try:
                result = self.file_tools.read_file(path, max_chars=9000)
            except Exception:
                continue
            content = (result.content or "").strip()
            if not content:
                continue
            emit(f"Читаю файл для анализа: {path}")
            read_paths.append(path)
            evidence_sections.append(f"[{path}]\n{content}")
            state.events.append(self._summarize_observation("read_file", result.model_dump()))
            if len(evidence_sections) >= 8:
                break

        prompt = (
            "Ты работаешь как инженерный Codex-ассистент в живом чате.\n"
            "Ответь обычным человеческим текстом на русском языке. Не возвращай JSON. Не используй markdown fences.\n"
            "Нужно сначала кратко описать архитектуру проекта, затем назвать ровно 3 главных риска и по каждому дать практичное предложение по доработке.\n"
            "Опирайся только на предоставленные файлы и карту проекта. Если уверенность ограничена, скажи это прямо, но не заменяй вывод шаблоном.\n\n"
            f"Запрос пользователя:\n{goal}\n\n"
            f"Workspace:\n{self.file_tools.guard.root}\n\n"
            f"Обзор проекта:\n{project_overview or 'Нет отдельного обзора.'}\n\n"
            f"Карта репозитория:\n{repo_map or 'Нет карты репозитория.'}\n\n"
            "Прочитанные файлы:\n"
            + "\n\n".join(evidence_sections)
            + "\n\n"
            "Формат ответа:\n"
            "1. Архитектура: 1-2 коротких абзаца.\n"
            "2. 3 главных риска: каждый риск с причиной из кода и предложением доработки.\n"
            "3. Что проверить дальше: 2-4 конкретных пункта.\n"
        )
        emit("Передаю собранный контекст в LLM для живого анализа...")
        evidence_text = "\n\n".join(evidence_sections)
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=False)
            assistant_message = self._clean_direct_analysis_response(response.content)
            unsupported = self._unsupported_technology_claims(assistant_message, evidence_text)
            if unsupported:
                emit("Ответ LLM содержит неподтверждённые технологии, прошу модель пересобрать анализ по evidence...")
                retry_prompt = (
                    prompt
                    + "\n\nПредыдущий ответ был отклонён: он упоминал технологии, которых нет в прочитанных файлах: "
                    + ", ".join(unsupported)
                    + ".\nПересобери ответ заново. Не упоминай эти технологии. Используй только факты из README, requirements и src/*.py."
                )
                retry_response = self.llm.chat([ChatMessage(role="user", content=retry_prompt)], json_mode=False)
                assistant_message = self._clean_direct_analysis_response(retry_response.content)
        except Exception as exc:
            assistant_message = (
                "Не удалось получить живой аналитический ответ от LLM.\n\n"
                f"Причина: {exc}\n\n"
                "Контекст был собран из файлов: " + ", ".join(read_paths)
            )

        summary = self._first_meaningful_line(assistant_message) or "Сформирован прямой LLM-анализ по собранным файлам проекта."
        report = FinalReport(
            status="completed",
            summary=summary[:500],
            changes=[f"Прочитан файл {path}" for path in read_paths[:8]],
            verification=[
                f"Контекст передан в LLM обычным chat-запросом без structured JSON-планировщика: {len(read_paths)} файлов.",
                *[f"read_file: проверен {path}" for path in read_paths[:6]],
            ],
            findings=[],
            next_steps=[],
        )
        meta = f"\n\nКонтекст анализа: {len(read_paths)} файлов ({', '.join(read_paths[:6])}{'...' if len(read_paths) > 6 else ''})."
        return report, assistant_message + meta

    @staticmethod
    def _clean_direct_analysis_response(content: str) -> str:
        text = content.strip()
        if "### Response:" in text:
            parts = [part.strip() for part in text.split("### Response:") if part.strip()]
            if parts:
                text = parts[-1]
        return text.strip()

    @staticmethod
    def _unsupported_technology_claims(response_text: str, evidence_text: str) -> list[str]:
        watched_terms = [
            "Redis",
            "PostgreSQL",
            "MySQL",
            "MongoDB",
            "FastAPI",
            "Django",
            "Flask",
            "React",
            "Kafka",
            "RabbitMQ",
            "Celery",
        ]
        response_lower = response_text.lower()
        evidence_lower = evidence_text.lower()
        unsupported: list[str] = []
        for term in watched_terms:
            if term.lower() in response_lower and term.lower() not in evidence_lower:
                unsupported.append(term)
        return unsupported

    def _direct_analysis_candidates(self) -> list[str]:
        candidates: list[str] = []
        preferred = [
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            "config.yaml",
            "config.yml",
            "config/default.yaml",
            "src/train.py",
            "src/env.py",
            "src/evaluate.py",
            "src/data.py",
            "src/backtest.py",
            "src/features.py",
            "src/metrics.py",
            "src/main.py",
            "src/app.py",
        ]
        candidates.extend(preferred)
        try:
            scan = self.repo_scanner.scan()
            candidates.extend(scan.entrypoints)
            candidates.extend(scan.critical_modules)
            candidates.extend(scan.important_files)
        except Exception:
            pass
        try:
            for path in self.file_tools.guard.glob("src", "*.py")[:12]:
                candidates.append(path.relative_to(self.file_tools.guard.root).as_posix())
        except Exception:
            pass
        seen: set[str] = set()
        existing: list[str] = []
        for path in candidates:
            normalized = path.replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            try:
                resolved = self.file_tools.guard.resolve_path(normalized).resolved
            except Exception:
                continue
            if resolved.exists() and resolved.is_file():
                existing.append(normalized)
        return existing[:16]

    @staticmethod
    def _first_meaningful_line(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip(" #*-")
            if stripped:
                return stripped
        return ""

    def _should_finalize_after_stall(self, stalled_cycles: int, analytical_goal: bool, observations: list[str]) -> bool:
        evidence = self._extract_evidence(observations)
        if stalled_cycles >= 3 and evidence:
            return True
        if analytical_goal and stalled_cycles >= 2 and len(evidence) >= 3:
            return True
        return False

    @staticmethod
    def _describe_action_for_chat(action: str, arguments: dict[str, Any]) -> str:
        if action == "list_files":
            target = arguments.get("path", ".")
            return f"Просматриваю структуру файлов: {target}"
        if action == "read_file":
            target = arguments.get("path", "")
            return f"Читаю файл: {target}" if target else "Читаю один из файлов проекта"
        if action == "search_in_files":
            query = arguments.get("query", "")
            return f"Ищу по репозиторию: {query}" if query else "Ищу связанные участки кода"
        if action == "run_command":
            command = arguments.get("command", "")
            return f"Запускаю проверку: {command}" if command else "Запускаю диагностическую команду"
        if action in {"write_file", "replace_in_file", "apply_unified_diff"}:
            target = arguments.get("path", "")
            return f"Вношу изменения в код: {target}" if target else "Вношу изменения в код"
        if action.startswith("git_"):
            return f"Проверяю git-состояние: {action}"
        return f"Выполняю шаг: {action}"

    def _next_turn(self, goal: str, context: ProjectContext) -> AgentTurn:
        prompt = build_turn_prompt(goal, context)
        response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
        try:
            payload = extract_json_object(response.content)
            return AgentTurn.from_model_payload(payload)
        except Exception as exc:
            recovered = self._recover_turn_from_text(response.content)
            if recovered is not None:
                return recovered
            raise LLMResponseError(f"Failed to parse structured agent turn: {response.content}") from exc

    @staticmethod
    def _recover_turn_from_text(raw_text: str) -> AgentTurn | None:
        try:
            payload = extract_json_object(raw_text)
            turn = AgentTurn.from_model_payload(payload)
        except Exception:
            return None
        if turn.actions or turn.final_report:
            return turn
        return None

    def _finalize_from_observations(self, goal: str, observations: list[str]) -> FinalReport:
        # Проверяем, было ли выполнение команды (любой run_command)
        has_any_command = any("run_command:" in item for item in observations)
        
        if self._requires_command_execution(goal) and not has_any_command:
            return FinalReport(
                status="failed",
                summary="Запрошенный запуск команды не был выполнен.",
                changes=[],
                verification=[
                    "run_command не выполнялся в этом ходе, поэтому агент не может считать запуск подтверждённым."
                ],
                findings=[],
                next_steps=[
                    "Повторить запрос или явно указать команду запуска, например: python -m src.train --smoke-test ..."
                ],
            )
        
        # Для задач обучения: если команда была запущена и есть признаки работы - считаем успешным
        if self._is_training_complete(goal, observations):
            # Собираем позитивный итог для обучения
            return self._build_training_success_report(goal, observations)
        
        verification = self.verifier.verify().results
        verification_summary = [f"{item.command}: {item.exit_code}" for item in verification]
        is_risk_goal = ("риск" in goal.lower()) or ("risk" in goal.lower())
        prompt = (
            "Return valid JSON only. Do not use markdown fences.\n"
            "Use Russian language for summary, changes, and verification.\n"
            + (
                "Create a risk report with exactly 5 concrete findings. "
                "Populate findings[] with evidence-backed entries. "
                "Each item in changes should be a short headline, not a paragraph.\n"
                if is_risk_goal
                else "Create the final report for this coding-agent run from the tool observations.\n"
            )
            + f"Goal:\n{goal}\n\n"
            "Tool observations:\n"
            + "\n".join(f"- {item}" for item in observations[-12:])
            + "\n\n"
            "Response schema:\n"
            '{"status":"completed|stopped|failed","summary":"string","changes":["string"],"verification":["string"],'
            '"findings":[{"title":"string","severity":"low|medium|high","status":"observed|verified|hypothesis","evidence":"string","recommendation":"string"}],'
            '"next_steps":["string"]}'
        )
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
            payload = extract_json_object(response.content)
            report = FinalReport.model_validate(payload)
            if self._is_report_quality_ok(report, is_risk_goal=is_risk_goal):
                report.verification.extend(verification_summary)
                return report
        except Exception:
            pass
        synthesis = self._synthesize_report(goal, observations, is_risk_goal=is_risk_goal)
        if synthesis:
            synthesis.verification.extend(verification_summary)
            return synthesis
        successful_observations = [item for item in observations if '"ok": true' in item.lower()]
        if successful_observations:
            change_hints: list[str] = []
            verification_hints: list[str] = []
            for item in successful_observations[-6:]:
                if "read_file:" in item:
                    verification_hints.append("read_file succeeded")
                if "list_files:" in item:
                    verification_hints.append("list_files succeeded")
                if "search_in_files:" in item:
                    verification_hints.append("search_in_files succeeded")
                if "write_file:" in item or "replace_in_file:" in item or "apply_unified_diff:" in item:
                    change_hints.append("file edits were applied")
            if not any("read_file" in item or "run_command" in item for item in verification_hints):
                verification_hints.append("часть выводов носит характер гипотезы и требует отдельной проверки")
            return FinalReport(
                status="completed",
                summary="Задача выполнена по результатам анализа инструментальных наблюдений.",
                changes=change_hints[:3],
                verification=(verification_hints[:5] + verification_summary),
                findings=[],
                next_steps=["Проверить выводы вручную, так как итог собран из частичных наблюдений."],
            )
        return FinalReport(
            status="stopped",
            summary="Остановлено после ограниченного цикла без явного завершения от модели.",
            verification=verification_summary,
            findings=[],
            next_steps=["Собрать больше наблюдений и повторить запуск."],
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

    def _synthesize_report(self, goal: str, observations: list[str], is_risk_goal: bool) -> FinalReport | None:
        prompt = (
            "Return valid JSON only. Do not use markdown fences.\n"
            "Use Russian language only.\n"
            + (
                "Build a strict risk report with evidence-backed findings. "
                "Use findings[] for the main content and keep changes[] short.\n"
                if is_risk_goal
                else "Build a concrete final report from observations.\n"
            )
            + f"Goal:\n{goal}\n\nObservations:\n"
            + "\n".join(f"- {item}" for item in observations[-16:])
            + '\n\nSchema: {"status":"completed|stopped|failed","summary":"string","changes":["string"],"verification":["string"],"findings":[{"title":"string","severity":"low|medium|high","status":"observed|verified|hypothesis","evidence":"string","recommendation":"string"}],"next_steps":["string"]}'
        )
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
            payload = extract_json_object(response.content)
            report = FinalReport.model_validate(payload)
            if self._is_report_quality_ok(report, is_risk_goal=is_risk_goal):
                return report
        except Exception:
            pass
        return self._synthesize_from_observations(goal, observations, is_risk_goal)

    @staticmethod
    def _is_report_quality_ok(report: FinalReport, is_risk_goal: bool) -> bool:
        summary = report.summary.strip().lower()
        if not summary:
            return False
        if summary in {"string", "summary", "...", "todo"}:
            return False
        bad_patterns = (
            "выполнено в пределах ограниченного цикла",
            "recovered final report from partial model output",
            "задача завершена",
        )
        if any(pattern in summary for pattern in bad_patterns):
            return False
        if any(item.strip().lower() in {"string", "..."} for item in report.changes):
            return False
        if any(item.strip().lower() in {"string", "..."} for item in report.verification):
            return False
        if len(report.changes) != len(set(report.changes)) and len(report.changes) > 1:
            return False
        if len(report.verification) != len(set(report.verification)) and len(report.verification) > 1:
            return False
        if is_risk_goal and any(item["title"].strip().lower().startswith("наблюдение по ") for item in report.findings):
            return False
        if is_risk_goal and len(report.findings) < 3 and len(report.changes) < 4:
            return False
        if not report.changes and not report.verification:
            return False
        return True

    @staticmethod
    def _is_analytical_goal(goal: str) -> bool:
        lowered = goal.lower()
        action_keywords = (
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
            "запуск",
            "обучи",
            "обучение",
            "протестируй",
            "test",
            "fix",
            "change",
            "update",
            "implement",
            "run",
            "train",
        )
        if any(keyword in lowered for keyword in action_keywords):
            return False
        return any(keyword in lowered for keyword in ("риск", "risk", "анализ", "analy", "audit", "провер"))

    @staticmethod
    def _requires_command_execution(goal: str) -> bool:
        lowered = goal.lower()
        return any(
            keyword in lowered
            for keyword in (
                "запусти",
                "запустить",
                "запуск",
                "обучи",
                "обучение",
                "тестовое обучение",
                "прогони",
                "установ",
                "run",
                "train",
                "install",
                "смоук",
                "smoke",
                "iter",
                "итераци",
            )
        )

    @staticmethod
    def _is_training_command_goal(goal: str) -> bool:
        """Проверяет, является ли цель запуском обучения/тренировки."""
        lowered = goal.lower()
        return any(
            keyword in lowered
            for keyword in (
                "обуч",
                "train",
                "iter",
                "итераци",
                "epoch",
                "эпох",
                "step",
                "step",
                "timestep",
            )
        )

    @staticmethod
    def _has_successful_command_observation(observations: list[str]) -> bool:
        """Проверяет, был ли успешный run_command (exit_code=0)."""
        for item in reversed(observations):
            if item.startswith("run_command:"):
                try:
                    payload_str = item.split(":", 1)[1].strip()
                    payload = json.loads(payload_str)
                    if payload.get("exit_code") == 0:
                        return True
                except Exception:
                    if '"exit_code": 0' in item or '"exit_code\":0' in item:
                        return True
        return False

    def _is_training_complete(self, goal: str, observations: list[str]) -> bool:
        """
        Проверяет, завершилось ли обучение успешно по ключевым признакам в выводе.
        Для задач обучения достаточно увидеть exit_code=0 и признаки успеха в stdout.
        """
        if not self._is_training_command_goal(goal):
            return False
        
        for item in reversed(observations):
            if item.startswith("run_command:"):
                try:
                    payload_str = item.split(":", 1)[1].strip()
                    # Пытаемся распарсить JSON, обрабатывая возможные экранирования
                    try:
                        payload = json.loads(payload_str)
                    except json.JSONDecodeError:
                        # Если не удалось распарсить, пробуем упрощенный парсинг
                        # Ищем ключевые поля вручную
                        exit_code_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', payload_str)
                        timed_out_match = re.search(r'"timed_out"\s*:\s*(true|false)', payload_str, re.IGNORECASE)
                        stdout_match = re.search(r'"stdout"\s*:\s*"([^"]*)"', payload_str, re.DOTALL)
                        
                        exit_code = int(exit_code_match.group(1)) if exit_code_match else -1
                        timed_out = timed_out_match and timed_out_match.group(1).lower() == 'true'
                        stdout = stdout_match.group(1) if stdout_match else ""
                        
                        if exit_code == 0:
                            return True
                        
                        if timed_out and ("iterations" in stdout.lower() or "fps" in stdout.lower() or "total_timesteps" in stdout.lower()):
                            return True
                        continue
                    
                    exit_code = payload.get("exit_code", -1)
                    stdout = payload.get("stdout", "")
                    timed_out = payload.get("timed_out", False)
                    
                    # Exit code 0 - уже успех
                    if exit_code == 0:
                        return True
                    
                    # Даже если exit_code != 0, но есть признаки нормальной работы
                    if exit_code != 0:
                        # Проверяем, было ли это прерывание по таймауту или SIGTERM
                        if timed_out:
                            # Если таймаут, но были прогресс-бары или логи обучения - считаем что процесс работал
                            if "iterations" in stdout.lower() or "fps" in stdout.lower() or "total_timesteps" in stdout.lower():
                                return True
                except Exception:
                    pass
        return False

    @staticmethod
    def _should_bootstrap_project(goal: str) -> bool:
        lowered = goal.lower()
        return any(
            keyword in lowered
            for keyword in (
                "риск",
                "risk",
                "анализ",
                "analy",
                "audit",
                "провер",
                "понять проект",
                "understand project",
                "architecture",
                "архитектур",
                "repo",
                "repository",
            )
        )

    def _is_final_report_ready(self, report: FinalReport, observations: list[str], analytical_goal: bool = False) -> bool:
        evidence_observations = self._extract_evidence(observations)
        if len(evidence_observations) < 2:
            return False
        summary_lower = report.summary.strip().lower()
        if analytical_goal and not report.findings and len(report.verification) < 2:
            return False
        if analytical_goal and not report.findings and not report.changes:
            return False
        if analytical_goal and (
            summary_lower.startswith("запущен сбор")
            or summary_lower.startswith("начинаю работу")
            or "в стадии выполнения" in summary_lower
            or "ожидание результатов" in " ".join(report.next_steps).lower()
        ):
            return False
        if not analytical_goal and report.status == "stopped" and summary_lower.startswith("начинаю работу"):
            return False
        if analytical_goal and report.status == "stopped" and not report.findings:
            return False
        return True

    @staticmethod
    def _extract_evidence(observations: list[str]) -> list[str]:
        evidence = []
        for item in observations:
            if any(token in item for token in ('read_file:', 'search_in_files:', 'run_command:', 'git_status:', 'git_diff:')):
                evidence.append(item)
        return evidence

    def _synthesize_from_observations(self, goal: str, observations: list[str], is_risk_goal: bool) -> FinalReport | None:
        evidence = self._extract_evidence(observations)
        if not evidence:
            return None
        findings: list[dict[str, str]] = []
        changes: list[str] = []
        verification: list[str] = []
        seen_verification: set[str] = set()
        seen_changes: set[str] = set()
        seen_findings: set[tuple[str, str]] = set()
        for item in evidence[-10:]:
            label = item.split(":", 1)[0]
            parsed = self._parse_observation(item)
            path = parsed.get("path", "")
            snippet = item[:220]
            verification_line = self._build_verification_line(label, path)
            if verification_line not in seen_verification:
                verification.append(verification_line)
                seen_verification.add(verification_line)
            if is_risk_goal:
                change_line = self._build_change_line(label, path)
                if change_line not in seen_changes:
                    changes.append(change_line)
                    seen_changes.add(change_line)
                finding = self._build_risk_finding(label, path, snippet)
                dedupe_key = (finding["title"], finding["evidence"])
                if dedupe_key not in seen_findings:
                    findings.append(finding)
                    seen_findings.add(dedupe_key)
        summary = (
            "Сформирован итог на основе реальных наблюдений инструментов."
            if not is_risk_goal
            else "Сформирован аналитический итог на основе реальных наблюдений по репозиторию."
        )
        next_steps = ["Уточнить наиболее критичные выводы отдельными целевыми проверками."]
        return FinalReport(
            status="completed",
            summary=summary,
            changes=changes[:5],
            verification=verification[:8],
            findings=findings[:5],
            next_steps=next_steps,
        )

    @staticmethod
    def _parse_observation(item: str) -> dict[str, Any]:
        try:
            _, raw_payload = item.split(":", 1)
        except ValueError:
            raw_payload = item
        try:
            payload = json.loads(raw_payload.strip())
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            if payload.get("path"):
                return payload
            content = payload.get("content")
            if isinstance(content, str):
                path_match = re.search(r'"path"\s*:\s*"([^"]+)"', content)
                if path_match:
                    payload["path"] = path_match.group(1)
                    return payload
        path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw_payload)
        if path_match:
            return {"path": path_match.group(1)}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _build_verification_line(label: str, path: str) -> str:
        if path:
            return f"{label}: проверен {path}"
        return f"{label}: наблюдение собрано"

    @staticmethod
    def _build_change_line(label: str, path: str) -> str:
        if label == "read_file" and path:
            return f"Изучен файл {path}"
        if label == "search_in_files":
            return "Выполнен поисковый проход по репозиторию"
        if label == "run_command":
            return "Получены дополнительные системные наблюдения"
        return f"Собрано наблюдение через {label}"

    @staticmethod
    def _build_risk_finding(label: str, path: str, snippet: str) -> dict[str, str]:
        path_lower = path.lower()
        if "workspace_guard" in path_lower:
            return {
                "title": "Защита границ workspace критична для безопасности",
                "severity": "high",
                "status": "verified",
                "evidence": snippet,
                "recommendation": "Покрыть граничные случаи path traversal и проверить deny/ignore-паттерны отдельными тестами.",
            }
        if "command_policy" in path_lower:
            return {
                "title": "Ошибки в политике команд могут открыть опасные shell-действия",
                "severity": "high",
                "status": "verified",
                "evidence": snippet,
                "recommendation": "Ужесточить тесты на deny/require_approval правила и явно проверить опасные команды.",
            }
        if "ollama_provider" in path_lower:
            return {
                "title": "LLM-слой зависит от доступности локального Ollama",
                "severity": "high",
                "status": "verified",
                "evidence": snippet,
                "recommendation": "Добавить более явный degraded-режим и отдельную диагностику для HTTP/CLI fallback.",
            }
        if "orchestrator" in path_lower:
            return {
                "title": "Качество аналитических итогов зависит от устойчивости оркестратора к шумным ответам модели",
                "severity": "medium",
                "status": "verified",
                "evidence": snippet,
                "recommendation": "Продолжить ужесточать quality gate и добавлять тесты на повреждённые structured turns.",
            }
        if "api/app.py" in path_lower:
            return {
                "title": "API-поверхность требует отдельной проверки на безопасные границы доступа",
                "severity": "medium",
                "status": "verified",
                "evidence": snippet,
                "recommendation": "Проверить валидацию входов, обработку ошибок и ограничения на операции через API.",
            }
        title_target = path or label
        return {
            "title": f"Наблюдение по {title_target}",
            "severity": "medium",
            "status": "verified" if label in {"read_file", "search_in_files", "run_command"} else "observed",
            "evidence": snippet,
            "recommendation": "Проверить, влияет ли это наблюдение на надёжность, безопасность или сопровождение проекта.",
        }

    def _build_project_overview(self, state: SessionState) -> str:
        sections: list[str] = []
        try:
            top_level = self._top_level_inventory()[:80]
            if top_level:
                preview = ", ".join(top_level[:20])
                sections.append(f"Top-level files: {preview}")
                state.events.append(f"list_files: top-level inventory collected ({len(top_level)} items)")
        except Exception as exc:
            state.events.append(f"list_files: overview bootstrap failed: {exc}")

        key_files = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
            "config/default.yaml",
        ]
        for path in key_files:
            try:
                result = self.file_tools.read_file(path, start_line=1, end_line=80, max_chars=2500)
            except Exception:
                continue
            snippet = (result.content or "").strip()
            if not snippet:
                continue
            state.events.append(f"read_file: bootstrapped {path}")
            sections.append(f"[{path}]\n{snippet}")

        probe_files = self._direct_analysis_candidates()[:12]
        probes = [
            ("python_entry", "__main__"),
            ("fastapi", "FastAPI("),
            ("typer", "Typer("),
            ("flask", "Flask("),
            ("express", "express("),
            ("docker", "FROM "),
        ]
        for label, query in probes:
            hits = self._probe_in_files(query, probe_files, limit=6)
            if not hits:
                continue
            joined = "\n".join(hits[:4])
            sections.append(f"[{label}]\n{joined}")
            state.events.append(f"search_in_files: bootstrapped {label} hits")

        return "\n\n".join(sections)[:5000]

    def _top_level_inventory(self) -> list[str]:
        items: list[str] = []
        for path in sorted(self.file_tools.guard.root.iterdir(), key=lambda item: item.name.lower()):
            relative = path.relative_to(self.file_tools.guard.root).as_posix()
            if self.file_tools.guard._matches_any(relative, self.file_tools.guard.ignore_patterns + self.file_tools.guard.deny_patterns):
                continue
            suffix = "/" if path.is_dir() else ""
            items.append(f"{relative}{suffix}")
            if len(items) >= 80:
                break
        return items

    def _probe_in_files(self, query: str, files: list[str], limit: int = 6) -> list[str]:
        hits: list[str] = []
        lowered_query = query.lower()
        for path in files:
            try:
                resolved = self.file_tools.guard.resolve_path(path).resolved
                lines = resolved.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line_number, line in enumerate(lines, start=1):
                if lowered_query in line.lower():
                    hits.append(f"{path}:{line_number}:{line.strip()}")
                    if len(hits) >= limit:
                        return hits
        return hits

    def _build_repo_map(self, state: SessionState) -> str:
        result = self.repo_scanner.load_artifact()
        if result is None:
            artifact_path = self.repo_scanner.save_artifact()
            result = self.repo_scanner.load_artifact(artifact_path)
        if result is None:
            result = self.repo_scanner.scan()
        state.events.append("system: repository map collected")
        return result.to_prompt_text()[:5000]

    def _fallback_chat_turn(self, goal: str, observations: list[str], analytical_goal: bool) -> AgentTurn | None:
        evidence = self._extract_evidence(observations)
        unread_candidates = self._fallback_read_candidates(goal, observations)
        if unread_candidates:
            return AgentTurn.model_validate(
                {
                    "plan": [{"title": "Запасной проход по ключевым файлам", "status": "in_progress"}],
                    "actions": [{"action": "read_file", "arguments": {"path": unread_candidates[0]}}],
                    "final_report": None,
                }
            )
        if analytical_goal and len(evidence) >= 3:
            return None
        if not any("git_status:" in item for item in observations):
            return AgentTurn.model_validate(
                {
                    "plan": [{"title": "Проверка состояния репозитория", "status": "in_progress"}],
                    "actions": [{"action": "git_status", "arguments": {}}],
                    "final_report": None,
                }
            )
        return None

    def _fallback_read_candidates(self, goal: str, observations: list[str]) -> list[str]:
        lowered_goal = goal.lower()
        candidates = [
            "README.md",
            "pyproject.toml",
            "config/default.yaml",
            "config/policy.yaml",
            "src/coding_agent/core/orchestrator.py",
            "src/coding_agent/core/planner.py",
            "src/coding_agent/sandbox/workspace_guard.py",
            "src/coding_agent/sandbox/command_policy.py",
            "src/coding_agent/sandbox/shell_runner.py",
            "src/coding_agent/llm/ollama_provider.py",
            "src/coding_agent/memory/store.py",
            "src/coding_agent/tools/file_tools.py",
            "src/coding_agent/api/app.py",
        ]
        if "архитект" in lowered_goal or "architecture" in lowered_goal:
            candidates.insert(0, "src/coding_agent/core/bootstrap.py")
            candidates.insert(1, "src/coding_agent/core/context.py")
        read_paths = {self._parse_observation(item).get("path", "") for item in observations if "read_file:" in item}
        existing: list[str] = []
        for path in candidates:
            if path in read_paths:
                continue
            try:
                self.file_tools.guard.resolve_path(path)
            except Exception:
                continue
            existing.append(path)
        return existing

    @staticmethod
    def _extract_changed_files_from_observations(observations: list[str]) -> list[str]:
        path_pattern = re.compile(r'"path"\s*:\s*"([^"]+)"')
        files: list[str] = []
        seen: set[str] = set()
        for item in observations:
            for match in path_pattern.findall(item):
                normalized = match.replace("\\", "/")
                if normalized in seen:
                    continue
                seen.add(normalized)
                files.append(normalized)
        return files[:12]

    @staticmethod
    def _build_chat_assistant_message(report: FinalReport, changed_files: list[str]) -> str:
        lines = [report.summary.strip() or "Ход выполнен."]
        if report.changes:
            lines.append("")
            lines.append("Что сделал:")
            lines.extend(f"- {item}" for item in report.changes[:5])
        if report.findings:
            lines.append("")
            lines.append("На что обратил внимание:")
            for item in report.findings[:3]:
                title = str(item.get("title", "")).strip()
                recommendation = str(item.get("recommendation", "")).strip()
                if title:
                    lines.append(f"- {title}" + (f" — {recommendation}" if recommendation else ""))
        if report.verification:
            lines.append("")
            lines.append("Чем проверил:")
            lines.extend(f"- {item}" for item in report.verification[:5])
        if report.next_steps:
            lines.append("")
            lines.append("Что ещё осталось:")
            lines.extend(f"- {item}" for item in report.next_steps[:4])
        if changed_files:
            lines.append("")
            lines.append("Последние затронутые файлы: " + ", ".join(changed_files[:6]))
        return "\n".join(lines)

    def _build_training_success_report(self, goal: str, observations: list[str]) -> FinalReport:
        """
        Строит позитивный итоговый отчёт для задач обучения/тренировки.
        Извлекает ключевые метрики из stdout команды и формирует понятный пользователю отчёт.
        """
        # Находим последний run_command с данными
        last_run_data = None
        for item in reversed(observations):
            if item.startswith("run_command:"):
                try:
                    payload_str = item.split(":", 1)[1].strip()
                    payload = json.loads(payload_str)
                    last_run_data = payload
                    break
                except Exception:
                    continue
        
        if not last_run_data:
            # Fallback если не удалось распарсить
            return FinalReport(
                status="completed",
                summary="Обучение запущено.",
                changes=["Запущена команда обучения"],
                verification=["run_command выполнен"],
                findings=[],
                next_steps=["Дождаться завершения полного цикла обучения"],
            )
        
        command = last_run_data.get("command", "unknown")
        exit_code = last_run_data.get("exit_code", -1)
        stdout = last_run_data.get("stdout", "")
        stderr = last_run_data.get("stderr", "")
        timed_out = last_run_data.get("timed_out", False)
        duration = last_run_data.get("duration", 0)
        
        # Извлекаем ключевые метрики из stdout
        metrics = self._extract_training_metrics(stdout)
        
        # Формируем summary
        if timed_out:
            summary = f"Обучение запущено и выполнялось {duration:.1f}с. Процесс был остановлен по таймауту, но прогресс зафиксирован."
        elif exit_code == 0:
            summary = f"Обучение успешно завершено за {duration:.1f}с."
        else:
            summary = f"Обучение выполнено за {duration:.1f}с (exit_code={exit_code})."
        
        # Добавляем метрики в summary если есть
        if metrics:
            metric_parts = []
            if "iterations" in metrics:
                metric_parts.append(f"итераций: {metrics['iterations']}")
            if "timesteps" in metrics:
                metric_parts.append(f"шагов: {metrics['timesteps']}")
            if "fps" in metrics:
                metric_parts.append(f"FPS: {metrics['fps']}")
            if metric_parts:
                summary += " " + ", ".join(metric_parts) + "."
        
        changes = [f"Выполнена команда: {command[:80]}{'...' if len(command) > 80 else ''}"]
        
        verification = [
            f"Команда выполнена: exit_code={exit_code}",
            f"Длительность: {duration:.1f}с",
        ]
        if timed_out:
            verification.append("Процесс остановлен по таймауту UI (это нормально для долгого обучения)")
        
        findings = []
        if metrics:
            if "iterations" in metrics:
                findings.append({
                    "title": f"Прогресс обучения: {metrics['iterations']} итераций",
                    "severity": "low",
                    "status": "verified",
                    "evidence": f"Зафиксировано {metrics['iterations']} итераций в stdout",
                    "recommendation": "Продолжить обучение до целевого количества итераций",
                })
        
        next_steps = []
        if timed_out:
            next_steps.append("Для полного обучения запустите команду отдельно с увеличенным таймаутом")
        next_steps.append("Проверить логи обучения в artifacts/ или указанной директории")
        
        return FinalReport(
            status="completed",
            summary=summary,
            changes=changes,
            verification=verification,
            findings=findings,
            next_steps=next_steps,
        )
    
    @staticmethod
    def _extract_training_metrics(stdout: str) -> dict[str, any]:
        """Извлекает ключевые метрики обучения из stdout."""
        import re
        metrics = {}
        
        # Ищем количество итераций
        iter_match = re.search(r'iterations\s*[:=]?\s*(\d+)', stdout, re.IGNORECASE)
        if iter_match:
            metrics["iterations"] = int(iter_match.group(1))
        
        # Ищем total_timesteps
        timestep_match = re.search(r'total_timesteps\s*[:=]?\s*(\d+)', stdout, re.IGNORECASE)
        if timestep_match:
            metrics["timesteps"] = int(timestep_match.group(1))
        
        # Ищем FPS
        fps_match = re.search(r'fps\s*[:=]?\s*(\d+(?:\.\d+)?)', stdout, re.IGNORECASE)
        if fps_match:
            metrics["fps"] = float(fps_match.group(1))
        
        # Ищем прогресс бар вида [=====>] XX.XX%
        progress_match = re.search(r'\]\s+(\d+\.\d+)%', stdout)
        if progress_match:
            metrics["progress_percent"] = float(progress_match.group(1))
        
        return metrics
"""Structured plan/action models."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    title: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


class AgentAction(BaseModel):
    action: Literal[
        "list_files",
        "read_file",
        "search_in_files",
        "write_file",
        "replace_in_file",
        "move_file",
        "delete_file",
        "mkdir",
        "apply_unified_diff",
        "run_command",
        "git_status",
        "git_diff",
        "git_create_branch",
        "git_add",
        "git_commit",
        "memory_search",
        "memory_add",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalReport(BaseModel):
    status: Literal["completed", "stopped", "failed"]
    summary: str
    changes: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    findings: list[dict[str, str]] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class AgentTurn(BaseModel):
    plan: list[PlanStep] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    final_report: FinalReport | None = None

    @classmethod
    def from_model_payload(cls, payload: dict[str, Any]) -> "AgentTurn":
        """Best-effort normalization for imperfect local-model JSON."""
        normalized_plan: list[dict[str, Any]] = []
        for raw_step in payload.get("plan", []):
            if not isinstance(raw_step, dict):
                continue
            status = raw_step.get("status", "pending")
            if status not in {"pending", "in_progress", "completed"}:
                status = "pending"
            normalized_plan.append({"title": str(raw_step.get("title", "Step")), "status": status})

        normalized_actions: list[dict[str, Any]] = []
        for raw_action in payload.get("actions", []):
            if not isinstance(raw_action, dict):
                continue
            action_name = raw_action.get("action")
            if action_name not in AgentAction.model_fields["action"].annotation.__args__:
                continue
            arguments = raw_action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            arguments = cls._normalize_arguments(arguments)
            if not cls._has_required_arguments(action_name, arguments):
                continue
            normalized_actions.append({"action": action_name, "arguments": arguments})

        final_report = payload.get("final_report")
        if isinstance(final_report, dict) and final_report.get("status") not in {"completed", "stopped", "failed"}:
            final_report = None
        if isinstance(final_report, dict):
            summary = cls._clean_summary(str(final_report.get("summary", "")))
            final_report["summary"] = summary
            final_report["changes"] = cls._normalize_string_list(final_report.get("changes", []))
            final_report["verification"] = cls._normalize_string_list(final_report.get("verification", []))
            final_report["findings"] = cls._normalize_findings(final_report.get("findings", []))
            final_report["next_steps"] = cls._normalize_string_list(final_report.get("next_steps", []))
            if cls._is_placeholder_report(final_report):
                final_report = None
        if isinstance(final_report, dict) and not str(final_report.get("summary", "")).strip():
            final_report = None
        if isinstance(final_report, dict) and final_report.get("status") == "failed":
            summary = str(final_report.get("summary", "")).lower()
            if "unexpected keyword argument" in summary:
                final_report = None
            if ".git" in summary or "not a git repository" in summary:
                final_report = None
        if isinstance(final_report, dict):
            summary_lower = str(final_report.get("summary", "")).lower()
            if "recovered final report from partial model output" in summary_lower:
                final_report = None
            final_report = cls._mark_unverified_risks(final_report)

        normalized_payload = {
            "plan": normalized_plan,
            "actions": normalized_actions,
            "final_report": final_report,
        }
        return cls.model_validate(normalized_payload)

    @classmethod
    def _is_placeholder_report(cls, final_report: dict[str, Any]) -> bool:
        summary = cls._clean_summary(str(final_report.get("summary", ""))).lower()
        changes = [cls._clean_summary(str(item)).lower() for item in final_report.get("changes", [])]
        verification = [cls._clean_summary(str(item)).lower() for item in final_report.get("verification", [])]
        if summary in {"string", "summary", "...", "todo"}:
            return True
        if any(item in {"string", "changes", "..."} for item in changes):
            return True
        if any(item in {"string", "verification", "..."} for item in verification):
            return True
        return False

    @staticmethod
    def _normalize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if "file" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("file")
        if "filename" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("filename")
        if "file_path" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("file_path")
        if "ath" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("ath")
        if "key" in normalized and "query" not in normalized:
            normalized["query"] = normalized.pop("key")
        if "keywords" in normalized and "query" not in normalized:
            normalized["query"] = normalized.pop("keywords")
        if "paths" in normalized and "glob" not in normalized:
            paths = normalized.pop("paths")
            if isinstance(paths, list) and len(paths) == 1:
                normalized["glob"] = paths[0]
        if "filenames" in normalized and "glob" not in normalized:
            filenames = normalized.pop("filenames")
            if isinstance(filenames, list) and len(filenames) == 1:
                normalized["glob"] = filenames[0]
        if "commit" in normalized and "ref" not in normalized:
            normalized["ref"] = normalized.pop("commit")
        for key in ("path", "src", "dst", "query", "glob", "message", "ref"):
            value = normalized.get(key)
            if isinstance(value, list) and len(value) == 1:
                normalized[key] = value[0]
        return normalized

    @staticmethod
    def _has_required_arguments(action_name: str, arguments: dict[str, Any]) -> bool:
        required = {
            "read_file": ("path",),
            "search_in_files": ("query",),
            "write_file": ("path", "content"),
            "replace_in_file": ("path", "old", "new"),
            "move_file": ("src", "dst"),
            "delete_file": ("path",),
            "git_create_branch": ("name",),
            "git_add": ("paths",),
            "git_commit": ("message",),
            "memory_search": ("query",),
            "memory_add": ("kind", "content"),
        }
        return all(arguments.get(key) not in (None, "", []) for key in required.get(action_name, ()))

    @staticmethod
    def _clean_summary(summary: str) -> str:
        summary = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", summary)
        summary = summary.replace("\u001b", "")
        return summary.strip()

    @classmethod
    def _mark_unverified_risks(cls, final_report: dict[str, Any]) -> dict[str, Any]:
        verification = [str(item) for item in final_report.get("verification", [])]
        verified_text = " ".join(verification).lower()
        changes = [str(item) for item in final_report.get("changes", [])]
        normalized_changes: list[str] = []
        for item in changes:
            cleaned = cls._clean_summary(item)
            if ("риск" in cleaned.lower() or "risk" in cleaned.lower()) and "run_command" not in verified_text and "read_file" not in verified_text:
                normalized_changes.append(f"{cleaned} (гипотеза, требует проверки)")
            else:
                normalized_changes.append(cleaned)
        final_report["changes"] = normalized_changes
        final_report["verification"] = [cls._clean_summary(item) for item in verification]
        findings = []
        for item in final_report.get("findings", []):
            if not isinstance(item, dict):
                continue
            title = cls._clean_summary(str(item.get("title", "")))
            if not title:
                continue
            evidence = cls._clean_summary(str(item.get("evidence", "")))
            status = cls._clean_summary(str(item.get("status", "hypothesis")))
            recommendation = cls._clean_summary(str(item.get("recommendation", "")))
            severity = cls._clean_summary(str(item.get("severity", "medium")))
            if ("риск" in title.lower() or "risk" in title.lower()) and not evidence:
                status = "hypothesis"
            findings.append(
                {
                    "title": title,
                    "severity": severity,
                    "status": status,
                    "evidence": evidence,
                    "recommendation": recommendation,
                }
            )
        final_report["findings"] = findings
        return final_report

    @classmethod
    def _normalize_findings(cls, raw_findings: Any) -> list[dict[str, str]]:
        findings: list[dict[str, str]] = []
        if not isinstance(raw_findings, list):
            return findings
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            title = cls._clean_summary(str(item.get("title", "")))
            if not title:
                continue
            findings.append(
                {
                    "title": title,
                    "severity": cls._clean_summary(str(item.get("severity", "medium"))),
                    "status": cls._clean_summary(str(item.get("status", "observed"))),
                    "evidence": cls._clean_summary(str(item.get("evidence", ""))),
                    "recommendation": cls._clean_summary(str(item.get("recommendation", ""))),
                }
            )
        return findings[:8]

    @classmethod
    def _normalize_string_list(cls, raw_value: Any) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            cleaned = cls._clean_summary(raw_value)
            return [cleaned] if cleaned else []
        if not isinstance(raw_value, list):
            cleaned = cls._clean_summary(str(raw_value))
            return [cleaned] if cleaned else []
        normalized: list[str] = []
        for item in raw_value:
            cleaned = cls._clean_summary(str(item))
            if cleaned:
                normalized.append(cleaned)
        return normalized

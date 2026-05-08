"""JSON helpers."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any


def to_pretty_json(value: Any) -> str:
    """Serialize a value for CLI output."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def extract_json_object(payload: str) -> dict[str, Any]:
    """Extract the first valid JSON object from raw model output."""
    stripped = payload.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    if "```" in payload:
        for chunk in payload.split("```"):
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                candidates.append(candidate)
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(payload[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            try:
                parsed = json.loads(_escape_invalid_json_backslashes(candidate))
            except JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    recovered = _recover_partial_agent_payload(payload)
    if recovered is not None:
        return recovered
    raise JSONDecodeError("No JSON object found in payload", payload, 0)


def _escape_invalid_json_backslashes(payload: str) -> str:
    """Repair common local-model JSON with unescaped Windows paths."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", payload)


def _recover_partial_agent_payload(payload: str) -> dict[str, Any] | None:
    """Best-effort recovery for duplicated/truncated local-model JSON."""
    action_pattern = re.compile(
        r'"action"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})',
        flags=re.DOTALL,
    )
    actions: list[dict[str, Any]] = []
    for match in action_pattern.finditer(payload):
        action_name = match.group(1)
        args_text = _escape_invalid_json_backslashes(match.group(2))
        try:
            arguments = json.loads(args_text)
        except JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            actions.append({"action": action_name, "arguments": arguments})

    # Fallback for heavily corrupted argument objects: recover action/path pairs.
    if not actions:
        loose_pattern = re.compile(r'"action"\s*:\s*"([^"]+)"(.*?)(?="action"|$)', flags=re.DOTALL)
        path_pattern = re.compile(r'"path"\s*:\s*"([^"]+)"')
        for match in loose_pattern.finditer(payload):
            action_name = match.group(1)
            chunk = match.group(2)
            arguments: dict[str, Any] = {}
            path_match = path_pattern.search(chunk)
            if path_match:
                arguments["path"] = path_match.group(1)
            actions.append({"action": action_name, "arguments": arguments})

    final_report_match = re.search(r'"final_report"\s*:\s*(\{.*\})', payload, flags=re.DOTALL)
    final_report: dict[str, Any] | None = None
    if final_report_match:
        try:
            candidate = _escape_invalid_json_backslashes(final_report_match.group(1))
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                final_report = parsed
        except JSONDecodeError:
            final_report = None

    if not actions and final_report is None:
        return None
    return {"plan": [], "actions": actions, "final_report": final_report}

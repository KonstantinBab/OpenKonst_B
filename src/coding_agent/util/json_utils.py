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
    candidates.extend(_extract_balanced_json_candidates(payload))

    prioritized: list[str] = []
    seen: set[str] = set()
    for candidate in reversed(candidates):
        marker = candidate.strip()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        prioritized.append(candidate)
    candidates = prioritized

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


def _extract_balanced_json_candidates(payload: str) -> list[str]:
    """Return balanced top-level JSON object slices from noisy model text."""
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(payload):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = payload[start_index : index + 1].strip()
                if any(token in candidate for token in ('"actions"', '"final_report"', '"plan"')):
                    candidates.append(candidate)
                start_index = None
    return candidates


def _escape_invalid_json_backslashes(payload: str) -> str:
    """Repair common local-model JSON with unescaped Windows paths."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", payload)


def _recover_partial_agent_payload(payload: str) -> dict[str, Any] | None:
    """Best-effort recovery for duplicated/truncated local-model JSON."""
    payload = payload.replace("\r\n", "\n")
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
        path_pattern = re.compile(r'"(?:path|ath)"\s*:\s*"([^"]+)"')
        for match in loose_pattern.finditer(payload):
            action_name = match.group(1)
            chunk = match.group(2)
            arguments: dict[str, Any] = {}
            path_match = path_pattern.search(chunk)
            if path_match:
                arguments["path"] = path_match.group(1)
            actions.append({"action": action_name, "arguments": arguments})
    if not actions:
        actions = _recover_actions_from_partial_chunks(payload)

    actions = _dedupe_actions(actions)

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
    if final_report is None:
        final_report = _recover_partial_final_report(payload)

    if not actions and final_report is None:
        return None
    return {"plan": [], "actions": actions, "final_report": final_report}


def _recover_partial_final_report(payload: str) -> dict[str, Any] | None:
    status_match = re.search(r'"status"\s*:\s*"(completed|stopped|failed)"', payload, flags=re.IGNORECASE)
    summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', payload, flags=re.DOTALL)
    if not status_match and not summary_match:
        return None

    status = (status_match.group(1).lower() if status_match else "completed")
    summary = summary_match.group(1).strip() if summary_match else "Recovered final report from partial model output."
    if not summary:
        summary = "Recovered final report from partial model output."

    changes = re.findall(r'"changes"\s*:\s*\[\s*"([^"]+)"', payload)
    if not changes:
        changes = re.findall(r'"changes"\s*:\s*"([^"]+)"', payload)
    verification = re.findall(r'"verification"\s*:\s*\[\s*"([^"]+)"', payload)

    return {
        "status": status,
        "summary": summary,
        "changes": changes[:10],
        "verification": verification[:10],
    }


def _recover_actions_from_partial_chunks(payload: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    allowed_actions = {
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
    }
    for action_name in sorted(allowed_actions, key=len, reverse=True):
        pattern = re.compile(rf'"action"\s*:\s*"{re.escape(action_name)}"(.*?)((?="action")|(?="final_report")|$)', flags=re.DOTALL)
        for match in pattern.finditer(payload):
            chunk = match.group(1)
            arguments: dict[str, Any] = {}
            for key in ("path", "query", "glob", "message", "src", "dst", "kind", "content", "old", "new", "ref", "name"):
                value_match = re.search(rf'"(?:{key}|{key[1:] if len(key) > 1 else key})"\s*:\s*"([^"]+)"', chunk)
                if value_match:
                    arguments[key] = value_match.group(1)
            paths_match = re.search(r'"paths"\s*:\s*\[\s*"([^"]+)"\s*\]', chunk)
            if paths_match:
                arguments["paths"] = [paths_match.group(1)]
            actions.append({"action": action_name, "arguments": arguments})
    if actions:
        return actions

    # Extra fallback for heavily duplicated local-model text where "arguments"
    # is truncated or misspelled but the action and path still appear nearby.
    for action_name in sorted(allowed_actions, key=len, reverse=True):
        pattern = re.compile(
            rf'"action"\s*:\s*"{re.escape(action_name)}".{{0,500}}?"(?:path|ath)"\s*:\s*"([^"]+)"',
            flags=re.DOTALL,
        )
        for match in pattern.finditer(payload):
            actions.append({"action": action_name, "arguments": {"path": match.group(1)}})
    if actions:
        return actions

    # Final fallback: if the model clearly attempted read_file actions, recover any
    # workspace-looking file paths in order and bind them to read_file.
    if '"action":"read_file"' in payload or '"action": "read_file"' in payload:
        path_like_pattern = re.compile(r'(src/[A-Za-z0-9_./-]+\.py|config/[A-Za-z0-9_./-]+\.yaml|README\.md|pyproject\.toml)')
        for match in path_like_pattern.finditer(payload):
            actions.append({"action": "read_file", "arguments": {"path": match.group(1)}})
    return actions


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in actions:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique[:12]

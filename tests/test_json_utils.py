from json import JSONDecodeError

import pytest

from coding_agent.core.planner import AgentTurn
from coding_agent.util.json_utils import extract_json_object


def test_extract_json_object_from_fenced_payload() -> None:
    payload = '```json\n{"final_report":{"status":"completed","summary":"ok","changes":[],"verification":[]}}\n```'
    parsed = extract_json_object(payload)
    assert parsed["final_report"]["status"] == "completed"


def test_extract_json_object_repairs_windows_paths_after_thinking_text() -> None:
    payload = (
        "Thinking...\n...done thinking.\n"
        '{"actions":[{"action":"read_file","arguments":{"file_path":"F:\\chrommm\\OpenKonst_B\\pyproject.toml"}}]}'
    )
    parsed = extract_json_object(payload)
    assert parsed["actions"][0]["arguments"]["file_path"] == "F:\\chrommm\\OpenKonst_B\\pyproject.toml"


def test_extract_json_object_recovers_from_duplicated_local_model_payload() -> None:
    payload = (
        '{"plan":[{"title":"Summarize Repository Architecture","status":"in_progress'
        'Architecture","status":"in_progress"}],"actions":[{"action":"read_file","ar'
        'chitecture","status":"in_progress}],"actions":[{"action":"read_file","arguments":'
        '{"path":"F:\\chrommm\\OpenKonst_B\\README.md"}},{"action":"read_file","arguments":'
        '{"path":"F:\\chrommm\\OpenKonst_B\\pyproject.toml"}}]}'
    )
    parsed = extract_json_object(payload)
    assert len(parsed["actions"]) >= 2
    assert parsed["actions"][0]["action"] == "read_file"


def test_extract_json_object_recovers_from_broken_arguments_chunk() -> None:
    payload = (
        '{"plan":[{"title":"Inspect repository structure and files","status":"pendin'
        'files","status":"pending"}],"actions":[{"action":"list_files","arguments":{'
        'files","status":"pendin"}],"actions":[{"action":"list_files","arguments":{"path":"F:\\chrommm\\OpenKonst_B"}},'
        '{"action":"read_file","arguments":{"path":"F:\\chrommm\\OpenKonst_B\\README.md"}}]}'
    )
    parsed = extract_json_object(payload)
    assert len(parsed["actions"]) >= 2
    assert any(item["action"] == "read_file" for item in parsed["actions"])


def test_extract_json_object_recovers_partial_final_report_from_corrupted_payload() -> None:
    payload = (
        '{"plan":[{"title":"Generate Final Report","status":"in_progress"}],"actions":[],"final_report":'
        '{"status":"completed","summary":"Repository inspection completed successfully.","changes":"data/agent_memory.db",'
        '"verification":["git_status","git_diff"]}}'
    )
    parsed = extract_json_object(payload)
    assert parsed["final_report"]["status"] == "completed"
    assert "Repository inspection completed successfully." in parsed["final_report"]["summary"]
    assert parsed["final_report"]["verification"][0] == "git_status"


def test_extract_json_object_raises_when_missing() -> None:
    with pytest.raises(JSONDecodeError):
        extract_json_object("not json")


def test_extract_json_object_recovers_actions_from_thinking_and_corrupted_json() -> None:
    payload = """Thinking...
The user wants me to analyze the project's key risks.
...done thinking.

{"plan":[{"title":"Анализ конфигурации и
безопасности","status":"in_progress"}],"actions":[{"action":"read_file","arguments"качества","status":pending"}],"actions":[{"action":"read_file","arguments":{"path":"config/policy.yaml"}},{"action":"read_file","arguments":{"path":"src/coding_agent/core/orchestrator.py"}},{"action":"read_file","arguments":{"path":"src/coding_agent/sandbox/workspace_guard.py"}}],"final_report":{"status":"in_progress","summary":"","changes":[""],"verification":[],"findings":[],"next_steps":["Получить политику безопасности"]}}"""
    parsed = extract_json_object(payload)
    assert len(parsed["actions"]) >= 3
    assert parsed["actions"][0]["action"] == "read_file"
    assert any(item["arguments"].get("path") == "config/policy.yaml" for item in parsed["actions"])


def test_extract_json_object_recovers_doctor_final_report_corrupted_turn() -> None:
    payload = """Thinking...
The user wants me to add a new command.
...done thinking.

{"plan":[{"title":"Просмотреть текущую структуру CLI в
main.py","status":"in_progress"}],"actions":[{"action":"read_file","arguments":{"задач","status":"pending"}],"actions":[{"action":"read_file","aruments":{"path":"src/coding_agent/cli/main.py"}}],"final_report":{"status":"pending","ath":"src/coding_agent/cli/main.py"}}],"final_report":{"status":"pending","summary":"","changes":[],"verification":[],"findings":[],"next_steps":[}}}"""
    parsed = extract_json_object(payload)
    assert any(item["action"] == "read_file" for item in parsed["actions"])
    assert any(item["arguments"].get("path") == "src/coding_agent/cli/main.py" for item in parsed["actions"])


def test_extract_json_object_recovers_paths_when_path_key_is_truncated() -> None:
    payload = """Thinking...
...done thinking.
{"actions":[{"action":"read_file","arguments":{"path":"src/coding_agent/api/app.py"}},{"action":"read_file","arguments":{"ath":"src/coding_agent/sandbox/workspace_guard.py"}},{"action":"read_file","aruments":{"ath":"src/coding_agent/llm/ollama_provider.py"}}],"final_report":{"status":"pending","summary":"","changes":[],"verification":[]}}"""
    parsed = extract_json_object(payload)
    turn = AgentTurn.from_model_payload(parsed)
    recovered_paths = [item.arguments.get("path") for item in turn.actions]
    assert "src/coding_agent/api/app.py" in recovered_paths
    assert "src/coding_agent/sandbox/workspace_guard.py" in recovered_paths
    assert "src/coding_agent/llm/ollama_provider.py" not in recovered_paths


def test_extract_json_object_prefers_last_balanced_agent_payload_after_thinking() -> None:
    payload = """Thinking...
Example schema:
{"plan":[{"title":"Пример","status":"pending"}],"actions":[],"final_report":{"status":"completed","summary":"example","changes":[],"verification":[]}}

...done thinking.

{
  "plan": [
    {
      "title": "Проверка механизма изоляции рабочего пространства",
      "status": "pending"
    }
  ],
  "actions": [
    {
      "action": "read_file",
      "arguments": {
        "path": "src/coding_agent/sandbox/workspace_guard.py"
      }
    },
    {
      "action": "read_file",
      "arguments": {
        "path": "config/policy.yaml"
      }
    }
  ],
  "final_report": {
    "status": "in_progress",
    "summary": "",
    "changes": [],
    "verification": [],
    "findings": [],
    "next_steps": []
  }
}"""
    parsed = extract_json_object(payload)
    turn = AgentTurn.from_model_payload(parsed)
    recovered_paths = [item.arguments.get("path") for item in turn.actions]
    assert "src/coding_agent/sandbox/workspace_guard.py" in recovered_paths
    assert "config/policy.yaml" in recovered_paths
    assert turn.final_report is None

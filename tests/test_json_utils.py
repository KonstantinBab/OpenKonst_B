from json import JSONDecodeError

import pytest

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


def test_extract_json_object_raises_when_missing() -> None:
    with pytest.raises(JSONDecodeError):
        extract_json_object("not json")

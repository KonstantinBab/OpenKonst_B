import httpx
import pytest

from coding_agent.llm.base import ChatMessage
from coding_agent.llm.ollama_provider import OllamaProvider
from coding_agent.util.errors import LLMResponseError


def _response(method: str, url: str, status_code: int, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request(method, url)
    if json_body is None:
        return httpx.Response(status_code, request=request)
    return httpx.Response(status_code, request=request, json=json_body)


def test_warmup_retries_until_chat_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://127.0.0.1:11434", "deepseek-coder:6.7b", retry_backoff_seconds=0.0)
    responses = iter(
        [
            _response("POST", "http://127.0.0.1:11434/api/chat", 503),
            _response("GET", "http://127.0.0.1:11434/api/tags", 200, {"models": [{"name": "deepseek-coder:6.7b"}]}),
            _response("POST", "http://127.0.0.1:11434/api/chat", 200, {"message": {"content": "OK"}}),
        ]
    )

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return next(responses)

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return next(responses)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    provider.warmup()


def test_warmup_reports_tags_503_details(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(
        "http://127.0.0.1:11434",
        "deepseek-coder:6.7b",
        retry_backoff_seconds=0.0,
        warmup_timeout_seconds=1,
    )
    clock = iter([0.0, 0.1, 1.1])

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: _response("POST", "http://127.0.0.1:11434/api/chat", 503))
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _response("GET", "http://127.0.0.1:11434/api/tags", 503))
    monkeypatch.setattr("coding_agent.llm.ollama_provider.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("coding_agent.llm.ollama_provider.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "coding_agent.llm.ollama_provider.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ollama")),
    )

    with pytest.raises(LLMResponseError) as exc_info:
        provider.warmup()

    assert "CLI fallback failed" in str(exc_info.value)


def test_chat_retries_after_503_by_waiting_for_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://127.0.0.1:11434", "deepseek-coder:6.7b", retry_backoff_seconds=0.0, max_retries=2)
    calls = {"warmup": 0, "post": 0}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls["post"] += 1
        if calls["post"] == 1:
            return _response("POST", url, 503)
        return _response("POST", url, 200, {"message": {"content": "done"}})

    def fake_warmup() -> None:
        calls["warmup"] += 1

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(provider, "warmup", fake_warmup)
    monkeypatch.setattr("coding_agent.llm.ollama_provider.time.sleep", lambda *_args, **_kwargs: None)

    response = provider.chat([ChatMessage(role="user", content="hello")])

    assert response.content == "done"
    assert calls["warmup"] == 1


def test_chat_falls_back_to_ollama_cli_after_503(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://127.0.0.1:11434", "deepseek-coder:6.7b", retry_backoff_seconds=0.0, max_retries=1)

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return _response("POST", url, 503)

    class FakeCompletedProcess:
        returncode = 0
        stdout = '{"final_report":{"status":"completed","summary":"ok","changes":[],"verification":[]}}'
        stderr = ""

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("coding_agent.llm.ollama_provider.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    response = provider.chat([ChatMessage(role="user", content="hello")], json_mode=True)

    assert '"summary":"ok"' in response.content
    assert response.raw["provider"] == "ollama_cli"


def test_warmup_falls_back_to_ollama_cli_after_503_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(
        "http://127.0.0.1:11434",
        "deepseek-coder:6.7b",
        retry_backoff_seconds=0.0,
        warmup_timeout_seconds=1,
    )
    clock = iter([0.0, 0.1, 1.1])

    class FakeCompletedProcess:
        returncode = 0
        stdout = "OK"
        stderr = ""

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: _response("POST", "http://127.0.0.1:11434/api/chat", 503))
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _response("GET", "http://127.0.0.1:11434/api/tags", 503))
    monkeypatch.setattr("coding_agent.llm.ollama_provider.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("coding_agent.llm.ollama_provider.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("coding_agent.llm.ollama_provider.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    provider.warmup()


def test_warmup_switches_to_cli_immediately_when_tags_also_503(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://127.0.0.1:11434", "deepseek-coder:6.7b", retry_backoff_seconds=0.0)
    calls = {"post": 0, "get": 0, "cli": 0}

    class FakeCompletedProcess:
        returncode = 0
        stdout = "OK"
        stderr = ""

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        calls["post"] += 1
        return _response("POST", "http://127.0.0.1:11434/api/chat", 503)

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        calls["get"] += 1
        return _response("GET", "http://127.0.0.1:11434/api/tags", 503)

    def fake_run(*args: object, **kwargs: object) -> FakeCompletedProcess:
        calls["cli"] += 1
        return FakeCompletedProcess()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("coding_agent.llm.ollama_provider.subprocess.run", fake_run)

    provider.warmup()
    provider.chat([ChatMessage(role="user", content="hello")])

    assert calls == {"post": 1, "get": 1, "cli": 2}
    assert provider.force_cli is True

"""Tests for AI Gateway client, settings migration, UI controls, and two-stage analysis pipeline."""
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
import pytest
import tkinter as tk

from toolrecap_v2.api_client import (
    APIError,
    OpenAICompatibleClient,
    ResponseDefectType,
    build_request_payload,
    estimate_request_size,
    classify_response_defect,
    parse_json_loose,
    SCANNER_TIMEOUT,
    SEASON_CONNECTION_TIMEOUT,
    FINALIZER_TIMEOUT,
    API_TEST_TIMEOUT,
)
from toolrecap_v2.narration import (
    NarrationError,
    RecapManifest,
    RecapSegment,
    _chunk_ranges,
    _compute_gateway_cache_key,
    _format_time,
    _load_gateway_cache,
    _save_gateway_cache,
    prepare_narration_for_video,
    validate_manifest,
)
from toolrecap_v2.paths import default_data_directory
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.settings import AppSettings, SettingsStore
from toolrecap_v2.ui import SettingsDialog


# ---------------------------------------------------------------------------
# 1. Settings Migration and Persistence Tests
# ---------------------------------------------------------------------------

def test_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.api_endpoint == "http://127.0.0.1:20128/v1"
    assert settings.api_key == ""
    assert settings.scanner_model == "sub"
    assert settings.scanner_thinking == "max"
    assert settings.finalizer_model == "prime"
    assert settings.finalizer_thinking == "high"
    assert settings.scanner_parallelism == 2
    assert settings.api_chunk_seconds == 300
    assert settings.gateway_enabled is True
    assert settings.transcription_provider == "local"
    assert settings.transcription_api_key == ""
    assert settings.transcription_base_url == ""


def test_settings_migration_from_legacy_stt(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    legacy_data = {
        "transcription_provider": "openai",
        "api_key": "sk-openai-stt-secret-key",
        "api_base_url": "https://api.openai.com/v1",
    }
    settings_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    store = SettingsStore(settings_file)
    settings = store.load()

    # Legacy STT key should migrate to transcription_api_key
    assert settings.transcription_api_key == "sk-openai-stt-secret-key"
    assert settings.transcription_base_url == "https://api.openai.com/v1"
    # Gateway fields should receive safe defaults
    assert settings.api_endpoint == "http://127.0.0.1:20128/v1"
    assert settings.api_key == ""
    assert settings.scanner_model == "sub"
    assert settings.finalizer_model == "prime"


def test_settings_bounds_clamping(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    raw_data = {
        "scanner_parallelism": 99,  # Should clamp to 4
        "api_chunk_seconds": 10,    # Should clamp to 60
    }
    settings_file.write_text(json.dumps(raw_data), encoding="utf-8")

    store = SettingsStore(settings_file)
    settings = store.load()

    assert settings.scanner_parallelism == 4
    assert settings.api_chunk_seconds == 60


def test_settings_save_and_load_roundtrip(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    store = SettingsStore(settings_file)

    orig = AppSettings(
        api_endpoint="http://192.168.1.50:20128/v1",
        api_key="my-secret-key-123",
        scanner_model="custom-scanner",
        scanner_thinking="medium",
        finalizer_model="custom-finalizer",
        finalizer_thinking="xhigh",
        scanner_parallelism=3,
        api_chunk_seconds=450,
        gateway_enabled=True,
        transcription_provider="openai",
        transcription_api_key="sk-another-stt-key",
        transcription_base_url="https://custom.openai.endpoint",
    )
    store.save(orig)

    loaded = store.load()
    assert loaded.api_endpoint == "http://192.168.1.50:20128/v1"
    assert loaded.api_key == "my-secret-key-123"
    assert loaded.scanner_model == "custom-scanner"
    assert loaded.scanner_thinking == "medium"
    assert loaded.finalizer_model == "custom-finalizer"
    assert loaded.finalizer_thinking == "xhigh"
    assert loaded.scanner_parallelism == 3
    assert loaded.api_chunk_seconds == 450
    assert loaded.gateway_enabled is True
    assert loaded.transcription_provider == "openai"
    assert loaded.transcription_api_key == "sk-another-stt-key"


# ---------------------------------------------------------------------------
# 2. API Client Retries, Auth, Error Sanitization, Cancellation Tests
# ---------------------------------------------------------------------------

def test_parse_json_loose() -> None:
    # 1. Plain json
    assert parse_json_loose('{"ok": true, "count": 1}') == {"ok": True, "count": 1}

    # 2. Markdown fenced json
    fenced = '```json\n{\n  "events": [{"start_ms": 0}]\n}\n```'
    assert parse_json_loose(fenced) == {"events": [{"start_ms": 0}]}

    # 3. Text with markdown fence without json tag
    fenced_raw = 'Here is your response:\n```\n{"ok": true}\n```\nEnjoy!'
    assert parse_json_loose(fenced_raw) == {"ok": True}

    # 4. Leading and trailing explanatory text
    messy = 'Explanation: {"result": "success"} thank you'
    assert parse_json_loose(messy) == {"result": "success"}

    # 5. Invalid json raises ValueError
    with pytest.raises(ValueError):
        parse_json_loose("This has no json at all")

    # 6. Non-dict json raises ValueError
    with pytest.raises(ValueError):
        parse_json_loose("[1, 2, 3]")


def test_api_client_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers: dict[str, str] = {}

    class MockResponse:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode("utf-8")

        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *args) -> None:
            pass

    def mock_urlopen(req, timeout=None):
        nonlocal captured_headers
        captured_headers = dict(req.headers)
        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # 1. With API key
    client_auth = OpenAICompatibleClient("http://localhost:20128/v1", api_key="secret-token-xyz")
    res = client_auth.chat_json(model="sub", system="test", user_text="test")
    assert res == {"ok": True}
    assert captured_headers.get("Authorization") == "Bearer secret-token-xyz"

    # 2. Without API key
    client_no_auth = OpenAICompatibleClient("http://localhost:20128/v1", api_key="")
    captured_headers.clear()
    res = client_no_auth.chat_json(model="sub", system="test", user_text="test")
    assert res == {"ok": True}
    assert "Authorization" not in captured_headers


def test_api_client_retry_cascade_drops_params(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts_bodies: list[dict[str, Any]] = []

    class MockResponse:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode("utf-8")

        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *args) -> None:
            pass

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        attempts_bodies.append(body)
        if len(attempts_bodies) == 1:
            # First attempt: reject response_format
            import urllib.error
            raise urllib.error.HTTPError(
                url="http://localhost:20128/v1",
                code=400,
                msg="response_format not supported",
                hdrs={},
                fp=None,
            )
        elif len(attempts_bodies) == 2:
            # Second attempt: reject reasoning_effort
            import urllib.error
            raise urllib.error.HTTPError(
                url="http://localhost:20128/v1",
                code=400,
                msg="reasoning_effort not supported",
                hdrs={},
                fp=None,
            )
        # Third attempt succeeds!
        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1", api_key="")
    res = client.chat_json(model="sub", thinking="max", system="test", user_text="test")

    assert res == {"ok": True}
    assert len(attempts_bodies) == 3
    # 1st attempt had both
    assert "response_format" in attempts_bodies[0]
    assert "reasoning_effort" in attempts_bodies[0]
    # 2nd attempt dropped response_format
    assert "response_format" not in attempts_bodies[1]
    assert "reasoning_effort" in attempts_bodies[1]
    # 3rd attempt dropped both
    assert "response_format" not in attempts_bodies[2]
    assert "reasoning_effort" not in attempts_bodies[2]


def test_settings_migration_custom_key_and_v1_retention(tmp_path: Path) -> None:
    # 1. Old V2 with non-sk key and no api_endpoint -> moves to transcription_api_key, clears api_key
    file1 = tmp_path / "old_v2.json"
    file1.write_text(json.dumps({
        "transcription_provider": "openai",
        "api_key": "custom-openai-compatible-key-abc",
    }), encoding="utf-8")
    s1 = SettingsStore(file1).load()
    assert s1.transcription_api_key == "custom-openai-compatible-key-abc"
    assert s1.api_key == ""
    assert s1.api_endpoint == "http://127.0.0.1:20128/v1"

    # 2. V1-style with api_endpoint -> retains api_key for gateway
    file2 = tmp_path / "v1_style.json"
    file2.write_text(json.dumps({
        "api_endpoint": "http://localhost:20128/v1",
        "api_key": "gateway-secret-token",
        "scanner_model": "sub",
        "finalizer_model": "prime",
    }), encoding="utf-8")
    s2 = SettingsStore(file2).load()
    assert s2.api_key == "gateway-secret-token"
    assert s2.api_endpoint == "http://localhost:20128/v1"


def test_api_client_error_sanitization_never_logs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error
    secret = "sk-super-secret-key-987654321"

    def mock_urlopen(req, timeout=None):
        fp = io.BytesIO(f'{{"error": "invalid key {secret}"}}'.encode("utf-8"))
        raise urllib.error.HTTPError(
            url="http://localhost:20128/v1",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=fp,
        )

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1", api_key=secret)
    with pytest.raises(APIError) as exc_info:
        client.chat_json(model="sub", system="test", user_text="test")

    err_msg = str(exc_info.value)
    assert secret not in err_msg
    assert "***" in err_msg


def test_api_client_cancellation() -> None:
    cancel_event = threading.Event()
    cancel_event.set()

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError, match="đã bị dừng"):
        client.chat_json(model="sub", system="test", user_text="test", cancel_event=cancel_event)


def test_api_client_test_method(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleClient("http://localhost:20128/v1")

    # Mock success
    monkeypatch.setattr(client, "chat_json", lambda **kwargs: {"ok": True})
    assert client.test("sub", "max") == "OK"

    # Mock unexpected response
    monkeypatch.setattr(client, "chat_json", lambda **kwargs: {"ok": False, "status": "fail"})
    assert "Phản hồi không mong đợi" in client.test("sub", "max")


# ---------------------------------------------------------------------------
# 2b. Mandatory Cases A-H and Phase Timeout Tests
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data if data is not None else {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def read(self) -> bytes:
        return json.dumps(self._data).encode("utf-8")

    def __enter__(self) -> "_MockResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _make_http_error(
    code: int,
    msg: str = "Error",
    body: str = "",
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    fp = io.BytesIO(body.encode("utf-8")) if body else None
    return urllib.error.HTTPError(
        url="http://localhost:20128/v1/chat/completions",
        code=code,
        msg=msg,
        hdrs=headers or {},
        fp=fp,
    )


def test_case_a_503_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case A: 503 recoverable error retries and succeeds on 2nd attempt."""
    attempts = 0
    delays_called: list[float] = []
    status_history: list[str] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _make_http_error(503, msg="Service Unavailable", body="temporary outage")
        return _MockResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    res = client.chat_json(
        model="sub",
        system="test",
        user_text="test",
        on_status=status_history.append,
        _sleeper=delays_called.append,
    )

    assert res == {"ok": True}
    assert attempts == 2
    assert delays_called == [5.0]
    assert len(status_history) == 1
    assert "không khả dụng/unavailable" in status_history[0]
    assert "Thử lại 1/3" in status_history[0]


def test_case_b_three_attempt_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case B: 3 attempts of recoverable error (e.g. 503) exhausts retries and raises APIError."""
    attempts = 0
    delays_called: list[float] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts
        attempts += 1
        raise _make_http_error(503, msg="Service Unavailable", body="backend down")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            _sleeper=delays_called.append,
        )

    assert attempts == 3
    assert delays_called == [5.0, 15.0]
    err = str(exc_info.value)
    assert "thất bại sau 3 lần thử" in err or "failed after 3 attempts" in err
    assert "HTTP 503" in err


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_case_c_fatal_status_no_retry(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    """Case C: 401/403/404 fatal errors fail immediately without retry or fallback."""
    attempts = 0
    delays_called: list[float] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts
        attempts += 1
        raise _make_http_error(status_code, msg="Auth/Not Found", body="fatal error")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            _sleeper=delays_called.append,
        )

    assert attempts == 1
    assert len(delays_called) == 0
    assert f"HTTP {status_code}" in str(exc_info.value)


def test_case_d_429_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case D: 429 rate limit respects Retry-After header delay."""
    attempts = 0
    delays_called: list[float] = []
    status_history: list[str] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _make_http_error(
                429,
                msg="Too Many Requests",
                body="rate limit exceeded",
                headers={"Retry-After": "7"},
            )
        return _MockResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    res = client.chat_json(
        model="sub",
        system="test",
        user_text="test",
        on_status=status_history.append,
        _sleeper=delays_called.append,
    )

    assert res == {"ok": True}
    assert attempts == 2
    assert delays_called == [7.0]
    assert len(status_history) == 1
    assert "quá tải/rate limited" in status_history[0]
    assert "Thử lại 1/3" in status_history[0]


def test_case_e_network_error_recovery_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case E: Network error (URLError/TimeoutError) retries and can recover or exhaust after 3 attempts."""
    # 1. Recovery
    attempts_rec = 0
    delays_rec: list[float] = []
    status_rec: list[str] = []

    def mock_urlopen_rec(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts_rec
        attempts_rec += 1
        if attempts_rec == 1:
            raise urllib.error.URLError("Connection refused")
        return _MockResponse({"choices": [{"message": {"content": '{"recovered": true}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_rec)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    res = client.chat_json(
        model="sub",
        system="test",
        user_text="test",
        on_status=status_rec.append,
        _sleeper=delays_rec.append,
    )
    assert res == {"recovered": True}
    assert attempts_rec == 2
    assert delays_rec == [5.0]
    assert len(status_rec) == 1
    assert "không khả dụng/unavailable" in status_rec[0]
    assert "Thử lại 1/3" in status_rec[0]

    # 2. Exhaustion (assert exactly 3 network attempts)
    attempts_exh = 0
    delays_exh: list[float] = []
    status_exh: list[str] = []

    def mock_urlopen_exh(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts_exh
        attempts_exh += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_exh)

    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            on_status=status_exh.append,
            _sleeper=delays_exh.append,
        )

    assert attempts_exh == 3
    assert delays_exh == [5.0, 15.0]
    assert len(status_exh) == 2
    assert "quá hạn kết nối/timeout" in status_exh[0]
    assert "quá hạn kết nối/timeout" in status_exh[1]
    assert "thất bại sau 3 lần thử" in str(exc_info.value) or "failed after 3 attempts" in str(exc_info.value)


def test_case_f_payload_fallback_independent_from_transport_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case F: 400/422 moves payload variant without consuming transport retry count."""
    attempts_bodies: list[dict[str, Any]] = []
    delays_called: list[float] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        body = json.loads(req.data.decode("utf-8"))
        attempts_bodies.append(body)
        idx = len(attempts_bodies)
        if idx == 1:
            # Variant 0: reject response_format with 400
            raise _make_http_error(400, msg="Bad Request", body="response_format not supported")
        elif idx == 2:
            # Variant 1: reject reasoning_effort with 422
            raise _make_http_error(422, msg="Unprocessable Entity", body="reasoning_effort not supported")
        elif idx == 3:
            # Variant 2, transport attempt 1: recoverable 503
            raise _make_http_error(503, msg="Service Unavailable", body="temp unavailable")
        elif idx == 4:
            # Variant 2, transport attempt 2: recoverable 503
            raise _make_http_error(503, msg="Service Unavailable", body="temp unavailable")
        elif idx == 5:
            # Variant 2, transport attempt 3: succeeds!
            return _MockResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})
        raise AssertionError("Unexpected request attempt")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    res = client.chat_json(
        model="sub",
        thinking="max",
        system="test",
        user_text="test",
        _sleeper=delays_called.append,
    )

    assert res == {"ok": True}
    assert len(attempts_bodies) == 5
    # Variant 0 had response_format and reasoning_effort
    assert "response_format" in attempts_bodies[0]
    assert "reasoning_effort" in attempts_bodies[0]
    # Variant 1 dropped response_format
    assert "response_format" not in attempts_bodies[1]
    assert "reasoning_effort" in attempts_bodies[1]
    # Variant 2 dropped both
    assert "response_format" not in attempts_bodies[2]
    assert "reasoning_effort" not in attempts_bodies[2]
    assert "response_format" not in attempts_bodies[3]
    assert "reasoning_effort" not in attempts_bodies[3]
    assert "response_format" not in attempts_bodies[4]
    assert "reasoning_effort" not in attempts_bodies[4]

    # Delays only called for the two 503s on variant 2 (transport retries), NOT for 400/422
    assert delays_called == [5.0, 15.0]


def test_case_g_cancellation_during_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case G: Cancellation event set during injected sleeper aborts request immediately."""
    # 1. Real threading.Event set during sleeper
    attempts_real = 0
    cancel_real = threading.Event()

    def sleeper_real(delay: float) -> None:
        cancel_real.set()

    def mock_urlopen_real(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts_real
        attempts_real += 1
        raise _make_http_error(503, msg="Service Unavailable", body="overloaded")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_real)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError, match="đã bị dừng"):
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            cancel_event=cancel_real,
            _sleeper=sleeper_real,
        )
    assert attempts_real == 1

    # 2. Custom cancellation event object set during sleeper
    attempts_custom = 0

    class CustomCancelEvent:
        def __init__(self) -> None:
            self._flag = False

        def set(self) -> None:
            self._flag = True

        def is_set(self) -> bool:
            return self._flag

    custom_event = CustomCancelEvent()

    def sleeper_custom(delay: float) -> None:
        custom_event.set()

    def mock_urlopen_custom(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        nonlocal attempts_custom
        attempts_custom += 1
        raise _make_http_error(503, msg="Service Unavailable", body="overloaded")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_custom)

    with pytest.raises(APIError, match="đã bị dừng"):
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            cancel_event=custom_event,  # type: ignore[arg-type]
            _sleeper=sleeper_custom,
        )
    assert attempts_custom == 1


def test_case_h_secret_safety_in_repr_callbacks_and_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case H: API secret key is never leaked in repr, error messages, status callbacks, or logs."""
    secret = "sk-super-secret-key-abcdef-999"
    client = OpenAICompatibleClient("http://localhost:20128/v1", api_key=secret)

    # 1. repr check
    client_repr = repr(client)
    assert secret not in client_repr
    assert "***" in client_repr

    # 2. HTTP error containing secret body, callbacks capture
    status_history: list[str] = []
    log_history: list[str] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        raise _make_http_error(
            500,
            msg="Internal Server Error",
            body=f"Crash for key {secret} with Bearer {secret}",
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            on_status=status_history.append,
            log=log_history.append,
            _sleeper=lambda d: None,
        )

    # Exception message must not contain secret
    err_str = str(exc_info.value)
    assert secret not in err_str
    assert "***" in err_str

    # Status callback must not contain secret
    assert len(status_history) > 0
    for st in status_history:
        assert secret not in st

    # Log callback must not contain secret
    assert len(log_history) > 0
    for lg in log_history:
        assert secret not in lg


def test_phase_timeouts_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase timeout mappings (scanner=300, season_connection=900, finalizer=900, api_test=120) and explicit override."""
    captured_timeouts: list[float | None] = []

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured_timeouts.append(timeout)
        return _MockResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1", timeout=60)

    # 1. scanner phase -> SCANNER_TIMEOUT (300s)
    client.chat_json(model="sub", system="t", user_text="t", phase="scanner")
    assert captured_timeouts[-1] == float(SCANNER_TIMEOUT)
    assert captured_timeouts[-1] == 300.0

    # 2. season_connection phase -> SEASON_CONNECTION_TIMEOUT (900s)
    client.chat_json(model="sub", system="t", user_text="t", phase="season_connection")
    assert captured_timeouts[-1] == float(SEASON_CONNECTION_TIMEOUT)
    assert captured_timeouts[-1] == 900.0

    # 3. finalizer phase -> FINALIZER_TIMEOUT (900s)
    client.chat_json(model="sub", system="t", user_text="t", phase="finalizer")
    assert captured_timeouts[-1] == float(FINALIZER_TIMEOUT)
    assert captured_timeouts[-1] == 900.0

    # 4. api_test phase -> API_TEST_TIMEOUT (120s)
    client.chat_json(model="sub", system="t", user_text="t", phase="api_test")
    assert captured_timeouts[-1] == float(API_TEST_TIMEOUT)
    assert captured_timeouts[-1] == 120.0

    # 5. subtitles / ocr phase -> 120s
    client.chat_json(model="sub", system="t", user_text="t", phase="ocr")
    assert captured_timeouts[-1] == 120.0

    # 6. explicit override overrides phase default
    client.chat_json(model="sub", system="t", user_text="t", phase="finalizer", timeout=42.0)
    assert captured_timeouts[-1] == 42.0

    # 7. no phase -> fallback to client.timeout (60s)
    client.chat_json(model="sub", system="t", user_text="t", phase=None)
    assert captured_timeouts[-1] == 60.0


# ---------------------------------------------------------------------------
# 3. Settings Dialog UI Controls and Test Buttons
# ---------------------------------------------------------------------------

def test_settings_dialog_instantiation_and_controls(tk_root: tk.Tk, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(
        api_endpoint="http://127.0.0.1:20128/v1",
        api_key="sk-test-pass",
        scanner_model="sub",
        scanner_thinking="max",
        finalizer_model="prime",
        finalizer_thinking="high",
        scanner_parallelism=2,
        api_chunk_seconds=300,
    )
    store = SettingsStore()

    dialog = SettingsDialog(tk_root, settings, store)
    try:
        # Verify UI controls instantiated
        assert dialog.endpoint_entry is not None
        assert dialog.key_entry is not None
        assert dialog.scanner_model_entry is not None
        assert dialog.scanner_thinking_combo is not None
        assert dialog.finalizer_model_entry is not None
        assert dialog.finalizer_thinking_combo is not None
        assert dialog.parallel_spin is not None
        assert dialog.chunk_spin is not None
        assert dialog.test_scanner_btn is not None
        assert dialog.test_finalizer_btn is not None
        assert dialog.ai_status_lbl is not None

        # Verify initial values
        assert dialog.endpoint_var.get() == "http://127.0.0.1:20128/v1"
        assert dialog.key_var.get() == "sk-test-pass"
        assert dialog.scanner_model_var.get() == "sub"
        assert dialog.finalizer_model_var.get() == "prime"
        assert dialog.parallel_var.get() == 2
        assert dialog.chunk_var.get() == 300

        # Test show/hide key toggle
        assert dialog.key_entry.cget("show") == "●"
        dialog.show_key_var.set(True)
        dialog._toggle_key()
        assert dialog.key_entry.cget("show") == ""
        dialog.show_key_var.set(False)
        dialog._toggle_key()
        assert dialog.key_entry.cget("show") == "●"

        # Test "Test Scanner" and "Test Finalizer" button handlers
        monkeypatch.setattr(OpenAICompatibleClient, "test", lambda self, m, t: "OK")

        dialog._test_api("scanner")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and "Scanner hoạt động" not in dialog.ai_status_var.get():
            dialog.update()
            time.sleep(0.02)
        assert "Scanner hoạt động" in dialog.ai_status_var.get()
        assert "OK" in dialog.ai_status_var.get()

        dialog._test_api("finalizer")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and "Finalizer hoạt động" not in dialog.ai_status_var.get():
            dialog.update()
            time.sleep(0.02)
        assert "Finalizer hoạt động" in dialog.ai_status_var.get()
        assert "OK" in dialog.ai_status_var.get()

    finally:
        try:
            dialog.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4. Chunking, Scanner/Finalizer Two-Stage Pipeline Success & Validation
# ---------------------------------------------------------------------------

def test_chunk_ranges_calculation() -> None:
    # Video duration 150s, chunk 60s
    ranges = _chunk_ranges(150.0, 60)
    assert len(ranges) == 3
    assert ranges[0] == (0.0, 60.0)
    assert ranges[1] == (60.0, 120.0)
    assert ranges[2] == (120.0, 150.0)

    # Video duration shorter than chunk
    ranges_short = _chunk_ranges(45.0, 300)
    assert len(ranges_short) == 1
    assert ranges_short[0] == (0.0, 45.0)


def test_gateway_two_stage_pipeline_success(tmp_path: Path, dummy_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock two-stage AI Gateway execution with valid scanner and finalizer outputs."""
    scan_called = []
    finalizer_called = []

    def mock_chat_json(self, *, model, system, user_text, **kwargs):
        if model == "sub":
            scan_called.append(user_text)
            return {
                "range_start_ms": 0,
                "range_end_ms": 2000,
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1500,
                        "summary": "Agent Carter finds the evidence file",
                        "dialogue_evidence": ["Agent Carter discovers"],
                    }
                ],
            }
        elif model == "prime":
            finalizer_called.append(user_text)
            return {
                "recap_title": "Episode 1 Recap",
                "segments": [
                    {
                        "segment_id": "scene_01",
                        "start_ms": 0,
                        "end_ms": 1500,
                        "narration_text": "Agent Carter discovers the crucial evidence file inside the archive.",
                        "audio_policy": "mute",
                    }
                ],
            }
        raise ValueError(f"Unknown model: {model}")

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_json)

    out_dir = tmp_path / "out_gateway"
    settings = AppSettings(
        api_endpoint="http://127.0.0.1:20128/v1",
        api_key="test-key",
        scanner_model="sub",
        finalizer_model="prime",
        api_chunk_seconds=300,
        gateway_enabled=True,
    )

    manifest = prepare_narration_for_video(dummy_video, out_dir, settings=settings)

    assert len(scan_called) >= 1
    assert len(finalizer_called) == 1
    assert len(manifest.segments) == 1
    assert manifest.segments[0].segment_id == "scene_01"
    assert manifest.segments[0].narration_text == "Agent Carter discovers the crucial evidence file inside the archive."
    assert manifest.segments[0].duration_ms == 1500


def test_gateway_malformed_and_out_of_bounds_validation(tmp_path: Path, dummy_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "out_err"
    settings = AppSettings(gateway_enabled=True)

    # 1. Scanner fails with API error
    def mock_scanner_fail(self, *args, **kwargs):
        raise APIError("Scanner rate limit exceeded")

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_scanner_fail)
    with pytest.raises(NarrationError, match="Scanner"):
        prepare_narration_for_video(dummy_video, out_dir, settings=settings)

    # 2. Finalizer returns no segments
    def mock_finalizer_empty(self, *, model, **kwargs):
        if model == "sub":
            return {"range_start_ms": 0, "range_end_ms": 2000, "events": []}
        return {"recap_title": "Empty", "segments": []}

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_finalizer_empty)
    with pytest.raises(NarrationError, match="danh sách phân đoạn"):
        prepare_narration_for_video(dummy_video, out_dir, settings=settings)

    # 3. Finalizer returns empty narration_text
    def mock_finalizer_empty_text(self, *, model, **kwargs):
        if model == "sub":
            return {"range_start_ms": 0, "range_end_ms": 2000, "events": []}
        return {
            "segments": [
                {"segment_id": "scene_01", "start_ms": 0, "end_ms": 1000, "narration_text": "   "}
            ]
        }

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_finalizer_empty_text)
    with pytest.raises(NarrationError, match="thiếu nội dung"):
        prepare_narration_for_video(dummy_video, out_dir, settings=settings)


# ---------------------------------------------------------------------------
# 5. Unreachable Gateway Causes Episode Error and UI State Restoration
# ---------------------------------------------------------------------------

def test_unreachable_gateway_causes_episode_error_and_ui_restored(tmp_path: Path, dummy_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_unreachable(self, *args, **kwargs):
        raise APIError("<urlopen error [Errno 111] Connection refused>")

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_unreachable)
    monkeypatch.setattr("toolrecap_v2.projects.transcribe_local_whisper", lambda *a, **kw: [(0.0, 1.0, "dummy dialog")])

    out_dir = tmp_path / "out_queue"
    settings = AppSettings(
        api_endpoint="http://127.0.0.1:20128/v1",
        gateway_enabled=True,
    )
    store = ProjectStore(tmp_path / "projects.json")

    record = ProjectRecord.from_video_path(dummy_video, out_dir)
    records = [record]
    store.save(records)

    ui_state_history: list[bool] = []

    def on_ui_state(is_running: bool) -> None:
        ui_state_history.append(is_running)

    queue = ProjectQueue(
        records,
        store=store,
        settings=settings,
        on_state_change=on_ui_state,
    )

    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=25)

    # Record must be marked ERROR
    assert record.status == "ERROR"
    assert "Không thể kết nối đến AI Gateway" in (record.error or "")

    # UI state must have transitioned from True (running) back to False (restored)
    assert ui_state_history == [True, False]
    assert queue.is_running is False


# ---------------------------------------------------------------------------
# 6. Gateway Cache Reuse Avoids API Calls
# ---------------------------------------------------------------------------

def test_gateway_cache_reuse_avoids_api_calls(tmp_path: Path, dummy_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_calls = 0

    # Isolate cache path from user LOCALAPPDATA
    monkeypatch.setattr("toolrecap_v2.narration.default_data_directory", lambda: tmp_path)

    def mock_chat_json(self, *, model, **kwargs):
        nonlocal api_calls
        api_calls += 1
        if model == "sub":
            return {"range_start_ms": 0, "range_end_ms": 2000, "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}]}
        return {
            "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1500, "narration_text": "Cached narration test"}
            ]
        }

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_json)

    out_dir = tmp_path / "out_cache"
    settings = AppSettings(
        api_endpoint="http://127.0.0.1:20128/v1",
        gateway_enabled=True,
    )

    # 1. First run: cold cache -> calls API
    manifest1 = prepare_narration_for_video(dummy_video, out_dir, settings=settings)
    assert api_calls >= 2
    initial_calls = api_calls

    # 2. Second run: warm cache -> must NOT call API again
    manifest2 = prepare_narration_for_video(dummy_video, out_dir, settings=settings)
    assert api_calls == initial_calls
    assert manifest2.segments[0].narration_text == manifest1.segments[0].narration_text


# ---------------------------------------------------------------------------
# 7. Sequential Batch Processing Unaffected
# ---------------------------------------------------------------------------

def test_sequential_batch_unaffected(tmp_path: Path, dummy_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure multiple projects in the queue execute strictly sequentially to completion."""
    # Provide companion manifest for dummy video so render proceeds deterministically
    companion = dummy_video.with_suffix(".json")
    companion.write_text(
        json.dumps({
            "project_id": "test_proj",
            "source_video": str(dummy_video),
            "recap_mode": "FULL_EPISODE",
            "recap_language": "en-US",
            "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "narration_text": "Scene 1"}
            ],
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_batch"
    store = ProjectStore(tmp_path / "batch_projects.json")

    rec1 = ProjectRecord.from_video_path(dummy_video, out_dir)
    rec2 = ProjectRecord.from_video_path(dummy_video, out_dir)
    records = [rec1, rec2]
    store.save(records)

    settings = AppSettings(gateway_enabled=True)
    queue = ProjectQueue(records, store=store, settings=settings)

    from tests.helpers_editorial import stage_response

    custom_output = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Batch Output",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 1.0}],
                        "narration": "Narration text",
                        "audio_policy": "duck",
                    }
                ],
            }
        ]
    }

    def mock_chat_json(self, *, model, **kwargs):
        if model == "sub":
            return {
                "range_start_ms": 0,
                "range_end_ms": 2000,
                "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
                "major_scenes": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
                "dialogue": [{"start_ms": 0, "end_ms": 1000, "speaker": "A", "quote": "Hi"}],
            }
        return stage_response(kwargs.get("system", ""), kwargs.get("user_text", ""), default=custom_output)
    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_json)

    from toolrecap_v2.renderer import PublicationRenderer
    from toolrecap_v2.domain.models import CommentaryOutput

    monkeypatch.setattr(
        PublicationRenderer,
        "render_manifest",
        lambda self, manifest, *a, **kw: [
            CommentaryOutput(
                output_id=o.output_id,
                title=o.title,
                publication_video_path=str(out_dir / f"{o.output_id}.mp4"),
                publication_original_srt_path=str(out_dir / f"{o.output_id}.original.srt"),
                publication_narration_srt_path=str(out_dir / f"{o.output_id}.narration.srt"),
                status="COMPLETED",
            )
            for o in manifest.outputs
        ],
    )

    # Fast-mock synthesize and render steps to test queue logic cleanly
    monkeypatch.setattr("toolrecap_v2.projects.extract_audio_from_video", lambda *a, **kw: True)
    monkeypatch.setattr("toolrecap_v2.projects.transcribe_local_whisper", lambda *a, **kw: [(0.0, 1.0, "dummy dialog")])
    monkeypatch.setattr("toolrecap_v2.projects.cut_clip", lambda *args, **kwargs: None)
    monkeypatch.setattr("toolrecap_v2.projects.run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr("toolrecap_v2.projects.render_final_video", lambda *args, **kwargs: None)
    monkeypatch.setattr("toolrecap_v2.projects.probe_duration", lambda p: 1.0)

    def mock_synth(self, voice_id, text, output_wav, **kwargs):
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(output_wav).write_bytes(b"RIFF" + b"\x00" * 40)

    monkeypatch.setattr("toolrecap_v2.voice.manager.VoiceModelManager.synthesize", mock_synth)

    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=10)

    assert rec1.status == "COMPLETED"
    assert rec2.status == "COMPLETED"
    assert queue.is_running is False


# ---------------------------------------------------------------------------
# 8. Defect Classification, HTTP 200 Retries, and Payload Estimator Tests
# ---------------------------------------------------------------------------

def test_response_defect_classification_all_six() -> None:
    """Classify all six response defect types, content list support, and valid responses."""
    # 1. Empty body
    assert classify_response_defect("") == ResponseDefectType.EMPTY_BODY
    assert classify_response_defect("   \n\t  ") == ResponseDefectType.EMPTY_BODY
    assert ResponseDefectType.classify("") == ResponseDefectType.EMPTY_BODY

    # 2. Outer JSON malformed / truncated / non-dict
    assert classify_response_defect("not json") == ResponseDefectType.OUTER_JSON_MALFORMED
    assert classify_response_defect('{"choices": ') == ResponseDefectType.OUTER_JSON_MALFORMED
    assert classify_response_defect("[1, 2, 3]") == ResponseDefectType.OUTER_JSON_MALFORMED
    assert classify_response_defect('"just a string"') == ResponseDefectType.OUTER_JSON_MALFORMED
    assert classify_response_defect("12345") == ResponseDefectType.OUTER_JSON_MALFORMED

    # 3. Choices absent / empty
    assert classify_response_defect(json.dumps({"id": "resp-123"})) == ResponseDefectType.CHOICES_MISSING
    assert classify_response_defect(json.dumps({"choices": []})) == ResponseDefectType.CHOICES_MISSING
    assert classify_response_defect(json.dumps({"choices": None})) == ResponseDefectType.CHOICES_MISSING
    assert classify_response_defect(json.dumps({"choices": "not a list"})) == ResponseDefectType.CHOICES_MISSING

    # 4. Message / content absent
    assert classify_response_defect(json.dumps({"choices": [{}]})) == ResponseDefectType.MESSAGE_CONTENT_MISSING
    assert classify_response_defect(json.dumps({"choices": [{"message": "not a dict"}]})) == ResponseDefectType.MESSAGE_CONTENT_MISSING
    assert classify_response_defect(json.dumps({"choices": [{"message": {}}]})) == ResponseDefectType.MESSAGE_CONTENT_MISSING
    assert classify_response_defect(json.dumps({"choices": [{"message": {"role": "assistant"}}]})) == ResponseDefectType.MESSAGE_CONTENT_MISSING

    # 5. Content empty
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": None}}]})) == ResponseDefectType.CONTENT_EMPTY
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": ""}}]})) == ResponseDefectType.CONTENT_EMPTY
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": "   \n\t  "}}]})) == ResponseDefectType.CONTENT_EMPTY
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": []}}]})) == ResponseDefectType.CONTENT_EMPTY
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": [{"type": "text", "text": "   "}]}}]})) == ResponseDefectType.CONTENT_EMPTY

    # 6. Malformed model JSON
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": "Just conversational text, no json"}}]})) == ResponseDefectType.MALFORMED_MODEL_JSON
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": "```json\n{broken: json\n```"}}]})) == ResponseDefectType.MALFORMED_MODEL_JSON
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": "[1, 2, 3]"}}]})) == ResponseDefectType.MALFORMED_MODEL_JSON

    # 7. Valid responses (string content and content list support)
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})) is None
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}]})) is None
    assert classify_response_defect(json.dumps({"choices": [{"message": {"content": [{"type": "text", "text": '{"ok": true}'}]}}]})) is None


@pytest.mark.parametrize("defect_body", [
    "",  # Empty body
    "<html>502 Bad Gateway</html>",  # Outer JSON malformed
    json.dumps({"id": "123"}),  # Choices missing
    json.dumps({"choices": [{}]}),  # Message missing
    json.dumps({"choices": [{"message": {"content": ""}}]}),  # Content empty
    json.dumps({"choices": [{"message": {"content": "Not valid JSON output"}}]}),  # Malformed model JSON
])
def test_http_200_defect_recovery_all_defects(monkeypatch: pytest.MonkeyPatch, defect_body: str) -> None:
    """HTTP 200 defect on attempt 1 retries with backoff and succeeds on attempt 2."""
    attempts = 0
    delays_called: list[float] = []
    status_history: list[str] = []

    class MockRawResponse:
        def __init__(self, raw: str) -> None:
            self._raw = raw

        def read(self) -> bytes:
            return self._raw.encode("utf-8")

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return MockRawResponse(defect_body)
        return MockRawResponse(json.dumps({"choices": [{"message": {"content": '{"recovered": true}'}}]}))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    res = client.chat_json(
        model="sub",
        system="test",
        user_text="test",
        on_status=status_history.append,
        _sleeper=delays_called.append,
    )

    assert res == {"recovered": True}
    assert attempts == 2
    assert delays_called == [5.0]
    assert len(status_history) == 1
    assert "Thử lại 1/3" in status_history[0]


@pytest.mark.parametrize("defect_body,expected_defect_name", [
    ("", "empty_body"),
    ("Not JSON content at all", "outer_json_malformed"),
    (json.dumps({"choices": []}), "choices_missing"),
    (json.dumps({"choices": [{"message": {}}]}), "message_content_missing"),
    (json.dumps({"choices": [{"message": {"content": "   "}}]}), "content_empty"),
    (json.dumps({"choices": [{"message": {"content": "invalid { json"}}]}), "malformed_model_json"),
])
def test_http_200_defect_exhaustion_all_defects(
    monkeypatch: pytest.MonkeyPatch, defect_body: str, expected_defect_name: str
) -> None:
    """3 consecutive attempts of HTTP 200 defect exhaust retries and raise APIError."""
    attempts = 0
    delays_called: list[float] = []

    class MockRawResponse:
        def __init__(self, raw: str) -> None:
            self._raw = raw

        def read(self) -> bytes:
            return self._raw.encode("utf-8")

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        nonlocal attempts
        attempts += 1
        return MockRawResponse(defect_body)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            _sleeper=delays_called.append,
        )

    assert attempts == 3
    assert delays_called == [5.0, 15.0]
    err = str(exc_info.value)
    assert "thất bại sau 3 lần thử" in err or "failed after 3 attempts" in err
    assert expected_defect_name in err


def test_http_200_defect_retries_same_variant_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 200 defect retries the SAME payload variant and does NOT fall back to next variant."""
    payloads_received: list[dict[str, Any]] = []
    delays_called: list[float] = []

    class MockRawResponse:
        def read(self) -> bytes:
            return b'{"choices": [{"message": {"content": "not json"}}]}'

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        payload = json.loads(req.data.decode("utf-8"))
        payloads_received.append(payload)
        return MockRawResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError):
        client.chat_json(
            model="sub",
            thinking="high",
            system="test",
            user_text="test",
            _sleeper=delays_called.append,
        )

    # Exactly 3 attempts made, all using variant 0 (response_format and reasoning_effort retained)
    assert len(payloads_received) == 3
    for p in payloads_received:
        assert p.get("response_format") == {"type": "json_object"}
        assert p.get("reasoning_effort") == "high"
    assert delays_called == [5.0, 15.0]


def test_http_200_defect_cancellation_during_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation during defect retry sleeper aborts immediately."""
    attempts = 0
    cancel_event = threading.Event()

    class MockRawResponse:
        def read(self) -> bytes:
            return b""  # Empty body defect

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def sleeper(delay: float) -> None:
        cancel_event.set()

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        nonlocal attempts
        attempts += 1
        return MockRawResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    with pytest.raises(APIError, match="đã bị dừng"):
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            cancel_event=cancel_event,
            _sleeper=sleeper,
        )

    assert attempts == 1


def test_domain_invalid_syntactic_dict_no_client_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid syntactic JSON dict that is domain-invalid returns once without client retry."""
    attempts = 0

    class MockRawResponse:
        def read(self) -> bytes:
            # Valid JSON object dict, but domain fields (e.g. segments/events) are absent
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({"unexpected_field": 123, "arbitrary": "data"})
                    }
                }]
            }).encode("utf-8")

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        nonlocal attempts
        attempts += 1
        return MockRawResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1")
    result = client.chat_json(model="sub", system="test", user_text="test")

    # Returned immediately on attempt 1 without retry
    assert attempts == 1
    assert result == {"unexpected_field": 123, "arbitrary": "data"}


def test_defect_error_sanitization_never_logs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secret API key in HTTP 200 defect response is never leaked in status, logs, or exceptions."""
    secret = "sk-defect-secret-key-12345"
    defect_raw = f'{{"choices": [{{"message": {{"content": "Error with key {secret} Bearer {secret}"}}}}]}}'
    status_history: list[str] = []
    log_history: list[str] = []

    class MockRawResponse:
        def read(self) -> bytes:
            return defect_raw.encode("utf-8")

        def __enter__(self) -> "MockRawResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: urllib.request.Request, timeout: float | None = None) -> MockRawResponse:
        return MockRawResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    client = OpenAICompatibleClient("http://localhost:20128/v1", api_key=secret)
    with pytest.raises(APIError) as exc_info:
        client.chat_json(
            model="sub",
            system="test",
            user_text="test",
            on_status=status_history.append,
            log=log_history.append,
            _sleeper=lambda d: None,
        )

    # Exception must not leak secret
    err_str = str(exc_info.value)
    assert secret not in err_str
    assert "***" in err_str

    # Status must not leak secret
    for st in status_history:
        assert secret not in st

    # Log must not leak secret
    for lg in log_history:
        assert secret not in lg


def test_estimator_exact_byte_size_and_build_payload(tmp_path: Path) -> None:
    """Payload builder and estimator exact byte calculation with Unicode, thinking, 3 variants, images."""
    # 1. Base test: exact byte match with manual serialization
    p0 = build_request_payload("sub", system="sys", user="usr", variant=0)
    expected_bytes0 = len(json.dumps(p0, ensure_ascii=False).encode("utf-8"))
    assert estimate_request_size("sub", system="sys", user="usr", variant=0) == expected_bytes0
    assert estimate_request_size(p0) == expected_bytes0

    # 2. Unicode: Vietnamese characters
    vn_sys = "Bạn là trợ lý AI phân tích phim."
    vn_user = "Đạo diễn: Trần Anh Hùng. Tóm tắt nội dung tập 1."
    p_vn = build_request_payload("sub", system=vn_sys, user=vn_user, variant=0)
    expected_vn_bytes = len(json.dumps(p_vn, ensure_ascii=False).encode("utf-8"))
    assert estimate_request_size("sub", system=vn_sys, user=vn_user, variant=0) == expected_vn_bytes

    # 3. Thinking levels
    # "auto" -> no reasoning_effort
    p_auto = build_request_payload("sub", system="sys", user="usr", thinking="auto", variant=0)
    assert "reasoning_effort" not in p_auto
    # "high" -> reasoning_effort="high"
    p_high = build_request_payload("sub", system="sys", user="usr", thinking="high", variant=0)
    assert p_high.get("reasoning_effort") == "high"
    assert estimate_request_size("sub", system="sys", user="usr", thinking="high", variant=0) == len(
        json.dumps(p_high, ensure_ascii=False).encode("utf-8")
    )

    # 4. All 3 variants:
    # Variant 0: response_format and reasoning_effort
    v0 = build_request_payload("sub", system="sys", user="usr", thinking="max", variant=0)
    assert "response_format" in v0
    assert "reasoning_effort" in v0
    assert estimate_request_size("sub", system="sys", user="usr", thinking="max", variant=0) == len(
        json.dumps(v0, ensure_ascii=False).encode("utf-8")
    )

    # Variant 1: reasoning_effort retained, response_format dropped
    v1 = build_request_payload("sub", system="sys", user="usr", thinking="max", variant=1)
    assert "response_format" not in v1
    assert "reasoning_effort" in v1
    assert estimate_request_size("sub", system="sys", user="usr", thinking="max", variant=1) == len(
        json.dumps(v1, ensure_ascii=False).encode("utf-8")
    )

    # Variant 2: both response_format and reasoning_effort dropped
    v2 = build_request_payload("sub", system="sys", user="usr", thinking="max", variant=2)
    assert "response_format" not in v2
    assert "reasoning_effort" not in v2
    assert estimate_request_size("sub", system="sys", user="usr", thinking="max", variant=2) == len(
        json.dumps(v2, ensure_ascii=False).encode("utf-8")
    )

    # 5. Image mocked file
    img_file = tmp_path / "mock_frame.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 32)
    p_img = build_request_payload("sub", system="sys", user="usr", images=[img_file], variant=0)
    user_content = p_img["messages"][1]["content"]
    assert len(user_content) == 2
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1]["image_url"]["detail"] == "high"
    assert estimate_request_size("sub", system="sys", user="usr", images=[img_file], variant=0) == len(
        json.dumps(p_img, ensure_ascii=False).encode("utf-8")
    )

    # 6. Data URL string directly
    data_url_str = "data:image/jpeg;base64,aGVsbG8gd29ybGQ="
    p_url = build_request_payload("sub", system="sys", user="usr", images=[data_url_str], variant=0)
    assert p_url["messages"][1]["content"][1]["image_url"]["url"] == data_url_str
    assert estimate_request_size("sub", system="sys", user="usr", images=[data_url_str], variant=0) == len(
        json.dumps(p_url, ensure_ascii=False).encode("utf-8")
    )

    # 7. Missing image file raises APIError
    missing_file = tmp_path / "does_not_exist.png"
    with pytest.raises(APIError, match="Thiếu file ảnh"):
        build_request_payload("sub", system="sys", user="usr", images=[missing_file])

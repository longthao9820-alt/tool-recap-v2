"""OpenAI-compatible AI Gateway client with loose JSON parsing, retry cascade, and secure key handling."""
from __future__ import annotations

import base64
import json
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error, request

SCANNER_TIMEOUT: int = 300
SEASON_CONNECTION_TIMEOUT: int = 900
FINALIZER_TIMEOUT: int = 900
API_TEST_TIMEOUT: int = 120

MAX_TRANSPORT_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (5.0, 15.0, 30.0)

RECOVERABLE_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
FATAL_HTTP_STATUSES: frozenset[int] = frozenset({401, 403, 404})
VARIANT_FALLBACK_HTTP_STATUSES: frozenset[int] = frozenset({400, 422})


class APIError(RuntimeError):
    """Raised when an AI Gateway request fails or returns an invalid payload."""
    pass


def parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON object loosely from response text, handling markdown code fences and wrappers."""
    value = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Không tìm thấy JSON object trong phản hồi.")
        result = json.loads(value[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Phản hồi API phải là một JSON object.")
    return result


def _data_url(path: Path) -> str:
    """Encode an image file to a base64 data URL."""
    mime = "image/png" if path.suffix.casefold() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _sanitize_error(message: str, secret: str) -> str:
    """Strip any accidental appearance of the API key from error messages."""
    if not message:
        return ""
    result = message
    if secret:
        result = result.replace(secret, "***")
    result = re.sub(r"(Bearer\s+)[^\s'\"]+", r"\1***", result)
    return result


def _resolve_timeout(phase: Any, explicit_timeout: float | None, default_timeout: float) -> float:
    """Resolve request timeout prioritizing explicit override, then phase mapping, then default."""
    if explicit_timeout is not None and explicit_timeout > 0:
        return float(explicit_timeout)
    if phase is not None:
        p = str(phase.value if hasattr(phase, "value") else phase).lower().strip()
        if p in ("scanner",):
            return float(SCANNER_TIMEOUT)
        if p in ("season_connection", "season_connecting"):
            return float(SEASON_CONNECTION_TIMEOUT)
        if p in ("finalizer", "season_mining"):
            return float(FINALIZER_TIMEOUT)
        if p in ("api_test", "test"):
            return float(API_TEST_TIMEOUT)
        if p in ("subtitles", "ocr"):
            return 120.0
    return float(default_timeout)


def _resolve_phase_label(phase: Any) -> str:
    """Resolve human-readable label for a phase."""
    if phase is None:
        return "AI Gateway"
    p = str(phase.value if hasattr(phase, "value") else phase).lower().strip()
    mapping = {
        "scanner": "Scanner",
        "season_connection": "Season Connection",
        "season_connecting": "Season Connection",
        "finalizer": "Finalizer",
        "season_mining": "Finalizer",
        "api_test": "API Test",
        "test": "API Test",
        "subtitles": "Subtitles",
        "ocr": "OCR Subtitles",
    }
    return mapping.get(p, str(phase))


def _extract_retry_after(exc: error.HTTPError) -> float | None:
    """Extract Retry-After header delay in seconds if present."""
    if not hasattr(exc, "headers") or not exc.headers:
        return None
    raw = None
    headers = exc.headers
    if hasattr(headers, "get"):
        raw = headers.get("Retry-After")
        if raw is None:
            raw = headers.get("retry-after")
    if raw is None and hasattr(headers, "items"):
        for k, v in headers.items():
            if str(k).lower() == "retry-after":
                raw = v
                break
    if raw is None:
        return None
    try:
        val = float(str(raw).strip())
        if val >= 0:
            return min(val, 120.0)
    except (ValueError, TypeError):
        pass
    return None


def _is_cancelled(cancel_event: Any) -> bool:
    """Check if cancellation event is set, supporting threading.Event and custom event objects."""
    if cancel_event is None:
        return False
    is_set_fn = getattr(cancel_event, "is_set", None)
    if callable(is_set_fn):
        return bool(is_set_fn())
    if is_set_fn is not None:
        return bool(is_set_fn)
    return False


def _wait_delay(
    delay: float,
    cancel_event: threading.Event | None,
    sleeper: Callable[[float], None] | None = None,
) -> None:
    """Wait for delay respecting cancel_event or custom sleeper."""
    if _is_cancelled(cancel_event):
        raise APIError("Yêu cầu API đã bị dừng.")

    if sleeper is not None:
        sleeper(delay)
        if _is_cancelled(cancel_event):
            raise APIError("Yêu cầu API đã bị dừng.")
        return

    if cancel_event is not None:
        signaled = cancel_event.wait(delay)
        if signaled or _is_cancelled(cancel_event):
            raise APIError("Yêu cầu API đã bị dừng.")
    else:
        if delay > 0:
            time.sleep(delay)


class OpenAICompatibleClient:
    """Dedicated client for OpenAI-compatible AI Gateways with retry cascade and phase handling."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        timeout: int = 120,
        base_url: str = "",
    ) -> None:
        raw_endpoint = endpoint or base_url
        self.endpoint = raw_endpoint.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout

    def __repr__(self) -> str:
        masked = "***" if self.api_key else ""
        return f"OpenAICompatibleClient(endpoint={self.endpoint!r}, api_key={masked!r}, timeout={self.timeout})"

    def chat_json(
        self,
        *,
        model: str,
        thinking: str = "auto",
        system: str,
        user_text: str,
        images: Iterable[Path] = (),
        max_tokens: int = 32_000,
        cancel_event: threading.Event | None = None,
        phase: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_delays: tuple[float, ...] | None = None,
        on_status: Callable[[str], None] | None = None,
        log: Callable[[str], None] | None = None,
        _sleeper: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request expecting JSON object response with payload fallback and retry cascade."""
        if _is_cancelled(cancel_event):
            raise APIError("Yêu cầu API đã bị dừng.")

        if not self.endpoint or not model.strip():
            raise APIError("Endpoint và model API không được để trống.")

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image in images:
            path = Path(image)
            if not path.is_file():
                raise APIError(f"Thiếu file ảnh: {path}")
            content.append({"type": "image_url", "image_url": {"url": _data_url(path), "detail": "high"}})

        base_payload: dict[str, Any] = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Payload variants: 1) full payload -> 2) drop response_format -> 3) drop reasoning_effort too
        variant_0 = dict(base_payload)
        variant_0["response_format"] = {"type": "json_object"}
        if thinking and thinking.casefold() != "auto":
            variant_0["reasoning_effort"] = thinking.casefold()

        variant_1 = dict(base_payload)
        if thinking and thinking.casefold() != "auto":
            variant_1["reasoning_effort"] = thinking.casefold()

        variant_2 = dict(base_payload)

        variants = [variant_0, variant_1, variant_2]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resolved_timeout = _resolve_timeout(phase, timeout, self.timeout)
        phase_label = _resolve_phase_label(phase)
        max_transport_attempts = max(1, max_retries if max_retries is not None else MAX_TRANSPORT_ATTEMPTS)
        delays = retry_delays if retry_delays is not None else DEFAULT_RETRY_DELAYS
        url = self.endpoint + "/chat/completions"

        last_error = ""

        for variant_idx, body in enumerate(variants):
            for attempt in range(1, max_transport_attempts + 1):
                if _is_cancelled(cancel_event):
                    raise APIError("Yêu cầu API đã bị dừng.")

                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                payload_bytes = len(encoded)
                api_request = request.Request(
                    url,
                    data=encoded,
                    headers=headers,
                    method="POST",
                )

                t0 = time.monotonic()
                try:
                    with request.urlopen(api_request, timeout=resolved_timeout) as response:
                        raw_text = response.read().decode("utf-8", "replace")
                    elapsed = time.monotonic() - t0
                    if log:
                        log(
                            _sanitize_error(
                                f"[AI Gateway] phase={phase_label} attempt={attempt}/{max_transport_attempts} "
                                f"model={model} timeout={resolved_timeout:.0f}s elapsed={elapsed:.2f}s "
                                f"payload_bytes={payload_bytes} -> 200 OK",
                                self.api_key,
                            )
                        )
                except error.HTTPError as exc:
                    elapsed = time.monotonic() - t0
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:3000]
                    except Exception:
                        detail = str(exc)
                    status_code = exc.code
                    raw_err = f"HTTP {status_code}: {detail}"
                    sanitized_err = _sanitize_error(raw_err, self.api_key)
                    last_error = sanitized_err

                    # 401, 403, 404: immediate fatal error (no retry, no fallback)
                    if status_code in FATAL_HTTP_STATUSES:
                        if log:
                            log(
                                _sanitize_error(
                                    f"[AI Gateway] phase={phase_label} attempt={attempt}/{max_transport_attempts} "
                                    f"fatal error: {sanitized_err}",
                                    self.api_key,
                                )
                            )
                        raise APIError(sanitized_err)

                    # 400, 422: parameter incompatibility -> fallback to next variant without consuming transport retry
                    if status_code in VARIANT_FALLBACK_HTTP_STATUSES:
                        if log:
                            log(
                                _sanitize_error(
                                    f"[AI Gateway] phase={phase_label} variant={variant_idx} got HTTP {status_code}. "
                                    f"Advancing payload variant without consuming transport retry.",
                                    self.api_key,
                                )
                            )
                        # Break inner loop to move to next variant
                        break

                    # 429, 500, 502, 503, 504: recoverable transport error
                    if status_code in RECOVERABLE_HTTP_STATUSES:
                        if attempt >= max_transport_attempts:
                            if log:
                                log(
                                    _sanitize_error(
                                        f"[AI Gateway] phase={phase_label} attempt={attempt}/{max_transport_attempts} "
                                        f"exhausted: {sanitized_err}",
                                        self.api_key,
                                    )
                                )
                            exhaust_msg = (
                                f"AI Gateway ({phase_label}) thất bại sau {max_transport_attempts} lần thử "
                                f"/ failed after {max_transport_attempts} attempts: {sanitized_err}"
                            )
                            raise APIError(_sanitize_error(exhaust_msg, self.api_key))

                        # Recoverable retry with backoff
                        retry_after = _extract_retry_after(exc)
                        if retry_after is not None:
                            delay = retry_after
                        else:
                            delay_idx = attempt - 1
                            delay = delays[delay_idx] if delay_idx < len(delays) else delays[-1]

                        if status_code == 429:
                            reason_vn = "quá tải"
                            reason_en = "rate limited"
                        else:
                            reason_vn = "không khả dụng"
                            reason_en = "unavailable"

                        retry_status = (
                            f"AI Gateway ({phase_label} - {reason_vn}/{reason_en}): "
                            f"Thử lại {attempt}/{max_transport_attempts}... "
                            f"/ Retrying {attempt}/{max_transport_attempts}..."
                        )
                        sanitized_status = _sanitize_error(retry_status, self.api_key)
                        if on_status:
                            on_status(sanitized_status)
                        if log:
                            log(
                                _sanitize_error(
                                    f"[AI Gateway] phase={phase_label} attempt {attempt}/{max_transport_attempts} failed "
                                    f"(HTTP {status_code}, elapsed={elapsed:.2f}s, bytes={payload_bytes}). "
                                    f"Retrying in {delay:.1f}s: {sanitized_err}",
                                    self.api_key,
                                )
                            )

                        _wait_delay(delay, cancel_event, _sleeper)
                        continue

                    # Any other HTTP status -> fatal
                    raise APIError(sanitized_err)

                except (error.URLError, TimeoutError, socket.timeout, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
                    elapsed = time.monotonic() - t0
                    raw_err = str(exc)
                    sanitized_err = _sanitize_error(raw_err, self.api_key)
                    last_error = sanitized_err

                    if attempt >= max_transport_attempts:
                        if log:
                            log(
                                _sanitize_error(
                                    f"[AI Gateway] phase={phase_label} attempt={attempt}/{max_transport_attempts} "
                                    f"exhausted: {sanitized_err}",
                                    self.api_key,
                                )
                            )
                        exhaust_msg = (
                            f"AI Gateway ({phase_label}) thất bại sau {max_transport_attempts} lần thử "
                            f"/ failed after {max_transport_attempts} attempts: {sanitized_err}"
                        )
                        raise APIError(_sanitize_error(exhaust_msg, self.api_key))

                    delay_idx = attempt - 1
                    delay = delays[delay_idx] if delay_idx < len(delays) else delays[-1]

                    is_timeout = (
                        isinstance(exc, (TimeoutError, socket.timeout))
                        or "timed out" in str(exc).lower()
                        or "timeout" in str(getattr(exc, "reason", "")).lower()
                    )
                    if is_timeout:
                        reason_vn = "quá hạn kết nối"
                        reason_en = "timeout"
                    else:
                        reason_vn = "không khả dụng"
                        reason_en = "unavailable"

                    retry_status = (
                        f"AI Gateway ({phase_label} - {reason_vn}/{reason_en}): "
                        f"Thử lại {attempt}/{max_transport_attempts}... "
                        f"/ Retrying {attempt}/{max_transport_attempts}..."
                    )
                    sanitized_status = _sanitize_error(retry_status, self.api_key)
                    if on_status:
                        on_status(sanitized_status)
                    if log:
                        log(
                            _sanitize_error(
                                f"[AI Gateway] phase={phase_label} attempt {attempt}/{max_transport_attempts} failed "
                                f"(network error, elapsed={elapsed:.2f}s, bytes={payload_bytes}). "
                                f"Retrying in {delay:.1f}s: {sanitized_err}",
                                self.api_key,
                            )
                        )

                    _wait_delay(delay, cancel_event, _sleeper)
                    continue

                # Successful HTTP 200 response
                try:
                    raw = json.loads(raw_text)
                    choices = raw.get("choices", [])
                    if not choices:
                        raise ValueError("Không tìm thấy choices trong phản hồi API.")
                    message = choices[0]["message"]["content"]
                    if isinstance(message, list):
                        message = "\n".join(
                            str(item.get("text", ""))
                            for item in message
                            if isinstance(item, dict) and (item.get("type") == "text" or "text" in item)
                        )
                    return parse_json_loose(message if isinstance(message, str) else str(message))
                except Exception as exc:
                    sanitized_text = _sanitize_error(raw_text[:4000], self.api_key)
                    raise APIError(f"Không đọc được JSON từ API: {exc}\n{sanitized_text}") from exc

        raise APIError(_sanitize_error(last_error or "Không thể kết nối API.", self.api_key))

    def test(
        self,
        model: str,
        thinking: str = "auto",
        *,
        cancel_event: threading.Event | None = None,
        on_status: Callable[[str], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> str:
        """Run a lightweight test ping to verify connectivity and model responsiveness."""
        result = self.chat_json(
            model=model,
            thinking=thinking,
            system="Return JSON only.",
            user_text='Return exactly {"ok": true}',
            max_tokens=500,
            phase="api_test",
            timeout=API_TEST_TIMEOUT,
            cancel_event=cancel_event,
            on_status=on_status,
            log=log,
        )
        return "OK" if result.get("ok") is True else f"Phản hồi không mong đợi: {result}"

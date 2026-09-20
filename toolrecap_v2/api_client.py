"""OpenAI-compatible AI Gateway client with loose JSON parsing, retry cascade, and secure key handling."""
from __future__ import annotations

import base64
from enum import Enum
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

HARD_PAYLOAD_CEILING: int = 500_000
TARGET_PAYLOAD_CEILING: int = 480_000
DEFAULT_ANALYSIS_PAYLOAD_CEILING: int = 500_000
DEFAULT_VISION_PAYLOAD_CEILING: int = 5_000_000


def resolve_payload_ceiling(phase: Any = None, max_payload_bytes: int | None = None) -> int:
    """Resolve maximum payload byte ceiling based on phase and explicit limit."""
    if max_payload_bytes is not None and max_payload_bytes > 0:
        return int(max_payload_bytes)
    if phase is not None:
        p = str(phase.value if hasattr(phase, "value") else phase).lower().strip()
        if any(k in p for k in ("vision", "ocr", "subtitles", "subtitle")):
            return DEFAULT_VISION_PAYLOAD_CEILING
    return DEFAULT_ANALYSIS_PAYLOAD_CEILING

MAX_TRANSPORT_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (5.0, 15.0, 30.0)

RECOVERABLE_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
FATAL_HTTP_STATUSES: frozenset[int] = frozenset({401, 403, 404})
VARIANT_FALLBACK_HTTP_STATUSES: frozenset[int] = frozenset({400, 422})


class APIError(RuntimeError):
    """Raised when an AI Gateway request fails or returns an invalid payload."""
    pass


class ResponseDefectType(str, Enum):
    """Six defect types that can occur in HTTP 200 AI Gateway responses."""
    EMPTY_BODY = "empty_body"
    OUTER_JSON_MALFORMED = "outer_json_malformed"
    CHOICES_MISSING = "choices_missing"
    MESSAGE_CONTENT_MISSING = "message_content_missing"
    CONTENT_EMPTY = "content_empty"
    MALFORMED_MODEL_JSON = "malformed_model_json"

    @classmethod
    def classify(cls, raw_text: str) -> ResponseDefectType | None:
        return classify_response_defect(raw_text)


# Aliases for enum members to ensure compatibility with varied naming
ResponseDefectType.CHOICES_EMPTY = ResponseDefectType.CHOICES_MISSING
ResponseDefectType.CHOICES_ABSENT = ResponseDefectType.CHOICES_MISSING
ResponseDefectType.MESSAGE_CONTENT_ABSENT = ResponseDefectType.MESSAGE_CONTENT_MISSING
ResponseDefectType.CONTENT_MISSING = ResponseDefectType.MESSAGE_CONTENT_MISSING
ResponseDefectType.MESSAGE_MISSING = ResponseDefectType.MESSAGE_CONTENT_MISSING
ResponseDefectType.OUTER_JSON = ResponseDefectType.OUTER_JSON_MALFORMED
ResponseDefectType.TRUNCATED_JSON = ResponseDefectType.OUTER_JSON_MALFORMED
ResponseDefectType.NON_DICT_JSON = ResponseDefectType.OUTER_JSON_MALFORMED
ResponseDefectType.MODEL_JSON_MALFORMED = ResponseDefectType.MALFORMED_MODEL_JSON


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


def _image_to_data_url(image: Path | str) -> str:
    """Convert an image file or data URL string to a base64 data URL."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    path = Path(image)
    if not path.is_file():
        raise APIError(f"Thiếu file ảnh: {path}")
    return _data_url(path)


def _extract_content_text(content: Any) -> str | None:
    """Extract string text from message content, supporting string or content parts list."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def build_request_payload(
    model: str,
    system: str = "",
    user: str = "",
    images: Iterable[Path | str] = (),
    thinking: str = "auto",
    max_tokens: int = 32_000,
    variant: int = 0,
    *,
    user_text: str = "",
) -> dict[str, Any]:
    """Build OpenAI-compatible request payload for a specific variant (0=full, 1=no response_format, 2=base)."""
    effective_user = user or user_text
    content: list[dict[str, Any]] = [{"type": "text", "text": effective_user}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img), "detail": "high"}})

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

    payload = dict(base_payload)
    if variant == 0:
        payload["response_format"] = {"type": "json_object"}
        if thinking and thinking.casefold() != "auto":
            payload["reasoning_effort"] = thinking.casefold()
    elif variant == 1:
        if thinking and thinking.casefold() != "auto":
            payload["reasoning_effort"] = thinking.casefold()
    # variant 2 or any other: base_payload without response_format or reasoning_effort

    return payload


def estimate_request_size(
    model: str | dict[str, Any],
    system: str = "",
    user: str = "",
    images: Iterable[Path | str] = (),
    thinking: str = "auto",
    max_tokens: int = 32_000,
    variant: int = 0,
    *,
    user_text: str = "",
) -> int:
    """Calculate exact byte size of serialized request payload."""
    if isinstance(model, dict):
        body = model
    else:
        body = build_request_payload(
            model=model,
            system=system,
            user=user or user_text,
            images=images,
            thinking=thinking,
            max_tokens=max_tokens,
            variant=variant,
        )
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def classify_response_defect(raw_text: str) -> ResponseDefectType | None:
    """Classify response defect across six categories or return None if valid."""
    if not raw_text or not raw_text.strip():
        return ResponseDefectType.EMPTY_BODY

    try:
        outer = json.loads(raw_text)
    except Exception:
        return ResponseDefectType.OUTER_JSON_MALFORMED

    if not isinstance(outer, dict):
        return ResponseDefectType.OUTER_JSON_MALFORMED

    choices = outer.get("choices")
    if choices is None or not isinstance(choices, list) or len(choices) == 0:
        return ResponseDefectType.CHOICES_MISSING

    first_choice = choices[0]
    if not isinstance(first_choice, dict) or "message" not in first_choice:
        return ResponseDefectType.MESSAGE_CONTENT_MISSING

    message = first_choice["message"]
    if not isinstance(message, dict) or "content" not in message:
        return ResponseDefectType.MESSAGE_CONTENT_MISSING

    content = message["content"]
    if content is None:
        return ResponseDefectType.CONTENT_EMPTY

    extracted = _extract_content_text(content)
    if extracted is None or not extracted.strip():
        return ResponseDefectType.CONTENT_EMPTY

    try:
        parse_json_loose(extracted)
    except Exception:
        return ResponseDefectType.MALFORMED_MODEL_JSON

    return None


def parse_api_response_body(raw_text: str) -> dict[str, Any]:
    """Parse JSON response text from AI Gateway, handling defects and markdown blocks."""
    defect = classify_response_defect(raw_text)
    if defect is not None:
        raise APIError(f"HTTP 200 defect ({defect.value}): {raw_text[:200]}")
    outer = json.loads(raw_text)
    content = outer["choices"][0]["message"]["content"]
    extracted = _extract_content_text(content)
    return parse_json_loose(extracted if isinstance(extracted, str) else str(extracted))


def _defect_reason(defect: ResponseDefectType) -> tuple[str, str]:
    """Return Vietnamese and English reason labels for a response defect."""
    mapping = {
        ResponseDefectType.EMPTY_BODY: ("phản hồi rỗng", "empty body"),
        ResponseDefectType.OUTER_JSON_MALFORMED: ("JSON không hợp lệ", "malformed outer JSON"),
        ResponseDefectType.CHOICES_MISSING: ("thiếu choices", "missing choices"),
        ResponseDefectType.MESSAGE_CONTENT_MISSING: ("thiếu nội dung phản hồi", "missing content"),
        ResponseDefectType.CONTENT_EMPTY: ("nội dung rỗng", "empty content"),
        ResponseDefectType.MALFORMED_MODEL_JSON: ("JSON mô hình không hợp lệ", "malformed model JSON"),
    }
    return mapping.get(defect, ("lỗi phản hồi", "response defect"))


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
        max_payload_bytes: int | None = None,
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

        variants = [
            build_request_payload(
                model=model,
                system=system,
                user=user_text,
                images=images,
                thinking=thinking,
                max_tokens=max_tokens,
                variant=v,
            )
            for v in (0, 1, 2)
        ]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resolved_timeout = _resolve_timeout(phase, timeout, self.timeout)
        phase_label = _resolve_phase_label(phase)
        payload_ceiling = resolve_payload_ceiling(phase, max_payload_bytes)
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
                if payload_bytes > payload_ceiling:
                    raise APIError(
                        _sanitize_error(
                            f"Kích thước yêu cầu ({payload_bytes} bytes) vượt quá giới hạn tối đa cho phép "
                            f"({payload_ceiling} bytes) cho phase '{phase_label}'. / "
                            f"Request payload size ({payload_bytes} bytes) exceeds maximum ceiling "
                            f"({payload_ceiling} bytes) for phase '{phase_label}'.",
                            self.api_key,
                        )
                    )
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

                # Successful HTTP 200 response handling
                try:
                    defect = classify_response_defect(raw_text)
                except Exception:
                    defect = ResponseDefectType.OUTER_JSON_MALFORMED

                if defect is None:
                    try:
                        outer = json.loads(raw_text)
                        content = outer["choices"][0]["message"]["content"]
                        extracted = _extract_content_text(content)
                        return parse_json_loose(extracted if isinstance(extracted, str) else str(extracted))
                    except Exception:
                        defect = ResponseDefectType.MALFORMED_MODEL_JSON

                sanitized_preview = _sanitize_error(raw_text, self.api_key)[:200]
                if len(raw_text) > 200:
                    sanitized_preview += "..."

                defect_reason_vn, defect_reason_en = _defect_reason(defect)
                sanitized_err = f"HTTP 200 defect ({defect.value}): {sanitized_preview}"
                last_error = sanitized_err

                if attempt >= max_transport_attempts:
                    if log:
                        log(
                            _sanitize_error(
                                f"[AI Gateway] phase={phase_label} attempt={attempt}/{max_transport_attempts} "
                                f"exhausted ({defect.value}): {sanitized_preview}",
                                self.api_key,
                            )
                        )
                    exhaust_msg = (
                        f"AI Gateway ({phase_label}) thất bại sau {max_transport_attempts} lần thử "
                        f"/ failed after {max_transport_attempts} attempts: {sanitized_err}"
                    )
                    raise APIError(_sanitize_error(exhaust_msg, self.api_key))

                # Recoverable defect retry with backoff on the SAME variant
                delay_idx = attempt - 1
                delay = delays[delay_idx] if delay_idx < len(delays) else delays[-1]

                retry_status = (
                    f"AI Gateway ({phase_label} - {defect_reason_vn}/{defect_reason_en}): "
                    f"Thử lại {attempt}/{max_transport_attempts}... "
                    f"/ Retrying {attempt}/{max_transport_attempts}..."
                )
                sanitized_status = _sanitize_error(retry_status, self.api_key)
                if on_status:
                    on_status(sanitized_status)
                if log:
                    log(
                        _sanitize_error(
                            f"[AI Gateway] phase={phase_label} attempt {attempt}/{max_transport_attempts} defect "
                            f"({defect.value}, elapsed={elapsed:.2f}s, bytes={payload_bytes}). "
                            f"Retrying in {delay:.1f}s: {sanitized_preview}",
                            self.api_key,
                        )
                    )

                _wait_delay(delay, cancel_event, _sleeper)
                continue

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

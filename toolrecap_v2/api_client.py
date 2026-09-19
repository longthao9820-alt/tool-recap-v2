"""OpenAI-compatible AI Gateway client with loose JSON parsing, retry cascade, and secure key handling."""
from __future__ import annotations

import base64
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request


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
    if secret and secret in message:
        return message.replace(secret, "***")
    return message


class OpenAICompatibleClient:
    """Dedicated client for OpenAI-compatible AI Gateways (e.g. Tool-REcap gateway at localhost)."""

    def __init__(self, endpoint: str, api_key: str = "", timeout: int = 120) -> None:
        self.endpoint = endpoint.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout

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
    ) -> dict[str, Any]:
        """Send chat completion request expecting JSON object response with retry cascade."""
        if cancel_event and cancel_event.is_set():
            raise APIError("Yêu cầu API đã bị dừng.")

        if not self.endpoint or not model.strip():
            raise APIError("Endpoint và model API không được để trống.")

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image in images:
            path = Path(image)
            if not path.is_file():
                raise APIError(f"Thiếu file ảnh: {path}")
            content.append({"type": "image_url", "image_url": {"url": _data_url(path), "detail": "high"}})

        payload: dict[str, Any] = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if thinking and thinking.casefold() != "auto":
            payload["reasoning_effort"] = thinking.casefold()

        # Retry cascade: 1) full payload -> 2) drop response_format -> 3) drop reasoning_effort too
        attempts = [
            payload,
            {key: value for key, value in payload.items() if key != "response_format"},
            {key: value for key, value in payload.items() if key not in {"response_format", "reasoning_effort"}},
        ]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = ""
        url = self.endpoint + "/chat/completions"

        for body in attempts:
            if cancel_event and cancel_event.is_set():
                raise APIError("Yêu cầu API đã bị dừng.")

            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            api_request = request.Request(
                url,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(api_request, timeout=self.timeout) as response:
                    raw_text = response.read().decode("utf-8", "replace")
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:3000]
                last_error = f"HTTP {exc.code}: {detail}"
                last_error = _sanitize_error(last_error, self.api_key)
                if exc.code in {401, 403, 404}:
                    break
                if exc.code in {400, 422}:
                    continue
                break
            except (error.URLError, TimeoutError, OSError) as exc:
                last_error = _sanitize_error(str(exc), self.api_key)
                break

            try:
                raw = json.loads(raw_text)
                message = raw["choices"][0]["message"]["content"]
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

    def test(self, model: str, thinking: str) -> str:
        """Run a lightweight test ping to verify connectivity and model responsiveness."""
        result = self.chat_json(
            model=model,
            thinking=thinking,
            system="Return JSON only.",
            user_text='Return exactly {"ok": true}',
            max_tokens=500,
        )
        return "OK" if result.get("ok") is True else f"Phản hồi không mong đợi: {result}"

"""Windows-safe deterministic title sanitization and collision resolution."""
from __future__ import annotations

import re
from typing import Sequence

RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

ILLEGAL_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_title(title: str, max_length: int = 120) -> str:
    """Sanitize title into a deterministic, Windows-safe filename string.

    - Replaces illegal characters (<>:"/\\|?* and control chars) safely.
    - Preserves letters, digits, dashes, and underscores.
    - Strips leading and trailing dots and spaces.
    - Avoids Windows reserved device names (CON, NUL, etc.).
    - Ensures length is at most max_length (default 120).
    """
    if not title:
        return "output"

    text = str(title).strip()

    # Friendly replacements
    text = text.replace(":", " - ")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    text = text.replace("|", "-")
    text = text.replace('"', "'")
    text = text.replace("?", "")
    text = text.replace("*", "")
    text = text.replace("<", "")
    text = text.replace(">", "")

    # Strip any remaining illegal or control chars
    text = ILLEGAL_CHARS_PATTERN.sub(" ", text)

    # Collapse repeated whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Strip Windows-unfriendly leading/trailing chars
    text = text.strip(" ._")

    if not text:
        text = "output"

    # Avoid reserved device names
    base_check = text.split(".")[0].upper()
    if base_check in RESERVED_NAMES:
        text = f"{text}_output"

    # Truncate to max_length
    if len(text) > max_length:
        text = text[:max_length].rstrip(" ._")

    if not text:
        text = "output"

    return text


def resolve_unique_titles(
    titles: Sequence[str],
    max_length: int = 120,
) -> list[str]:
    """Deterministically sanitize a list of titles and resolve collisions with numeric suffixes.

    Ensures every resulting title is Windows-safe, unique in the list, and <= max_length.
    """
    seen: set[str] = set()
    result: list[str] = []

    for raw in titles:
        base = sanitize_title(raw, max_length=max_length)
        if base not in seen:
            seen.add(base)
            result.append(base)
            continue

        # Collision resolution
        suffix_index = 1
        while True:
            suffix = f"_{suffix_index}"
            allowed_len = max_length - len(suffix)
            candidate_base = base[:allowed_len].rstrip(" ._")
            candidate = f"{candidate_base}{suffix}"
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
                break
            suffix_index += 1

    return result

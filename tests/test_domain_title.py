"""Tests for Windows-safe title sanitization and collision resolution."""
from __future__ import annotations

from toolrecap_v2.domain.title import resolve_unique_titles, sanitize_title


def test_sanitize_title_removes_illegal_characters() -> None:
    raw = 'Episode 1: "The Beginning" / Part 1? <Special> | *Uncensored*'
    sanitized = sanitize_title(raw)
    assert ":" not in sanitized
    assert '"' not in sanitized
    assert "/" not in sanitized
    assert "?" not in sanitized
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert "|" not in sanitized
    assert "*" not in sanitized
    assert sanitized == "Episode 1 - 'The Beginning' - Part 1 Special - Uncensored"


def test_sanitize_title_length_limit_120() -> None:
    long_title = "A" * 150
    sanitized = sanitize_title(long_title, max_length=120)
    assert len(sanitized) <= 120
    assert sanitized == "A" * 120


def test_sanitize_title_strips_trailing_dots_and_spaces() -> None:
    raw = "  ...Dangerous Title...   "
    sanitized = sanitize_title(raw)
    assert not sanitized.startswith(".")
    assert not sanitized.startswith(" ")
    assert not sanitized.endswith(".")
    assert not sanitized.endswith(" ")
    assert sanitized == "Dangerous Title"


def test_sanitize_title_windows_reserved_names() -> None:
    for name in ["CON", "prn", "AUX", "NUL", "COM1", "LPT1"]:
        sanitized = sanitize_title(name)
        assert sanitized.upper() not in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        assert sanitized.startswith(name) or sanitized.startswith(name.upper())


def test_resolve_unique_titles_handles_duplicates_and_length() -> None:
    titles = [
        "Episode 1: The Fight",
        "Episode 1: The Fight",
        "Episode 1: The Fight",
    ]
    unique = resolve_unique_titles(titles, max_length=120)
    assert len(unique) == 3
    assert unique[0] == "Episode 1 - The Fight"
    assert unique[1] == "Episode 1 - The Fight_1"
    assert unique[2] == "Episode 1 - The Fight_2"


def test_resolve_unique_titles_handles_collisions_at_length_boundary() -> None:
    long_base = "B" * 119
    titles = [long_base, long_base]
    unique = resolve_unique_titles(titles, max_length=120)
    assert len(unique) == 2
    assert len(unique[0]) <= 120
    assert len(unique[1]) <= 120
    assert unique[0] != unique[1]
    assert unique[1].endswith("_1")

"""Version information and semantic version parsing for ToolRecap V2."""
from __future__ import annotations

import re

__version__ = "0.3.2"
GITHUB_REPO = "longthao9820-alt/tool-recap-v2"


def parse_version(v: str) -> tuple[tuple[int, ...], int, str]:
    """Parse version string into comparable tuple: ((major, minor, patch), release_flag, prerelease).
    release_flag: 1 for standard release, 0 for prerelease.
    """
    clean = str(v).strip().lstrip("vV")
    if not clean:
        return ((0, 0, 0), 0, "")

    match = re.match(r"^(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+))?$", clean)
    if not match:
        return ((0, 0, 0), 1, "")

    nums_part = match.group(1)
    prerelease_part = match.group(2) or ""

    nums = tuple(int(x) for x in nums_part.split("."))
    while len(nums) < 3:
        nums = nums + (0,)

    release_flag = 0 if prerelease_part else 1
    return (nums, release_flag, prerelease_part)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.
    Returns:
       1 if v1 > v2
      -1 if v1 < v2
       0 if v1 == v2
    """
    p1 = parse_version(v1)
    p2 = parse_version(v2)

    if p1[0] != p2[0]:
        return 1 if p1[0] > p2[0] else -1

    if p1[1] != p2[1]:
        return 1 if p1[1] > p2[1] else -1

    if p1[2] != p2[2]:
        return 1 if p1[2] > p2[2] else -1

    return 0


def is_newer_version(current: str, candidate: str) -> bool:
    """Return True if candidate is strictly newer than current."""
    return compare_versions(candidate, current) > 0

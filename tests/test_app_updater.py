"""Tests for app updater: version comparison, zip-slip security, staged apply, rollback, and restart."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
import pytest

from toolrecap_v2.updater import (
    ReleaseAsset,
    ReleaseInfo,
    UpdateSecurityError,
    generate_apply_script,
    parse_release_payload,
    safe_extract_zip,
    select_portable_asset,
    verify_zip_safety,
)
from toolrecap_v2.version import compare_versions, is_newer_version, parse_version


def test_version_parsing_and_comparison() -> None:
    assert parse_version("0.1.0") == ((0, 1, 0), 1, "")
    assert parse_version("v0.2.1") == ((0, 2, 1), 1, "")
    assert parse_version("1.0.0-rc1") == ((1, 0, 0), 0, "rc1")

    assert is_newer_version("0.1.0", "0.2.0") is True
    assert is_newer_version("0.1.0", "v0.1.1") is True
    assert is_newer_version("0.2.0", "0.1.9") is False
    assert is_newer_version("0.1.0", "0.1.0") is False

    assert compare_versions("0.1.0", "0.2.0") == -1
    assert compare_versions("0.2.0", "0.1.0") == 1
    assert compare_versions("v0.1.0", "0.1.0") == 0


def test_select_portable_asset() -> None:
    assets = [
        ReleaseAsset(name="installer.exe", url="https://example.com/installer.exe"),
        ReleaseAsset(name="ToolRecapV2-v0.2.0-windows-portable.zip", url="https://example.com/portable.zip"),
        ReleaseAsset(name="source.tar.gz", url="https://example.com/source.tar.gz"),
    ]
    selected = select_portable_asset(assets)
    assert selected is not None
    assert selected.name == "ToolRecapV2-v0.2.0-windows-portable.zip"


def test_parse_release_payload() -> None:
    payload = {
        "tag_name": "v0.2.0",
        "name": "ToolRecap V2 - Update 0.2.0",
        "body": "- Faster render\n- Bug fixes",
        "html_url": "https://github.com/longthao9820-alt/tool-recap-v2/releases/tag/v0.2.0",
        "published_at": "2026-09-19T12:00:00Z",
        "assets": [
            {
                "name": "ToolRecapV2-portable.zip",
                "browser_download_url": "https://example.com/portable.zip",
                "size": 50000000,
                "content_type": "application/zip",
            },
            {
                "name": "ToolRecapV2-portable.zip.sha256.txt",
                "browser_download_url": "https://example.com/portable.zip.sha256.txt",
                "size": 64,
                "content_type": "text/plain",
            },
        ],
    }
    info = parse_release_payload(payload)
    assert info.version == "0.2.0"
    assert info.selected_asset is not None
    assert info.selected_asset.name == "ToolRecapV2-portable.zip"
    assert info.sha256_asset is not None


def test_zip_safety_rejects_zip_slip(tmp_path: Path) -> None:
    # 1. Relative path traversal (..)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../windows/system32/cmd.exe", b"malicious")
    buf.seek(0)

    with pytest.raises(UpdateSecurityError, match="zip-slip"):
        verify_zip_safety(buf)

    # 2. Absolute path
    buf_abs = io.BytesIO()
    with zipfile.ZipFile(buf_abs, "w") as zf:
        zf.writestr("/etc/passwd", b"malicious")
    buf_abs.seek(0)

    with pytest.raises(UpdateSecurityError, match="tuyệt đối"):
        verify_zip_safety(buf_abs)


def test_safe_extract_zip(tmp_path: Path) -> None:
    zip_file = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("app/ToolRecapV2.exe", b"binary")
        zf.writestr("app/runtime/notes.txt", b"notes")

    target_dir = tmp_path / "extracted"
    safe_extract_zip(zip_file, target_dir)

    assert (target_dir / "app" / "ToolRecapV2.exe").is_file()
    assert (target_dir / "app" / "runtime" / "notes.txt").read_bytes() == b"notes"


def test_generate_apply_script(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    target = tmp_path / "app"
    script = generate_apply_script(staged, target, exe_name="ToolRecapV2.exe")

    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "ToolRecapV2.exe" in content
    assert "BACKUP=" in content
    assert "xcopy" in content
    assert "rollback" in content.lower()
    assert ":wait_exit" in content
    assert "rmdir /s /q" in content


def test_extract_expected_sha256() -> None:
    from toolrecap_v2.updater import extract_expected_sha256, UpdateValidationError

    sha_sample = (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  other.zip\n"
        "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0 *ToolRecapV2-portable.zip\n"
    )
    extracted = extract_expected_sha256(sha_sample, "ToolRecapV2-portable.zip")
    assert extracted == "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"

    # Single line bare hash format (exactly one token)
    single = "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
    assert extract_expected_sha256(single, "ToolRecapV2-portable.zip") == single

    # Single line with whitespace around bare hash
    assert extract_expected_sha256(f"  {single} \n", "any.zip") == single

    # Single line named checksum with wrong filename -> must NOT accept bare hash
    with pytest.raises(UpdateValidationError, match="không chứa chữ ký khớp"):
        extract_expected_sha256(f"{single}  other.zip", "missing.zip")

    # Multi-line missing target filename
    with pytest.raises(UpdateValidationError, match="không chứa chữ ký khớp"):
        extract_expected_sha256(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  other.zip\n"
            "ffffc44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  another.zip\n",
            "missing.zip",
        )

    # Empty content
    with pytest.raises(UpdateValidationError, match="rỗng"):
        extract_expected_sha256("", "file.zip")


def test_download_and_stage_update_requires_sha256(tmp_path: Path) -> None:
    from toolrecap_v2.updater import download_and_stage_update, UpdateValidationError

    release = ReleaseInfo(
        tag_name="v0.2.0",
        version="0.2.0",
        name="Test",
        body="",
        html_url="",
        published_at="",
        selected_asset=ReleaseAsset(name="app.zip", url="http://example.com/app.zip"),
        sha256_asset=None,  # Missing required sha256
    )

    with pytest.raises(UpdateValidationError, match="thiếu tệp mã băm SHA256"):
        download_and_stage_update(release, staging_dir=tmp_path / "stage")

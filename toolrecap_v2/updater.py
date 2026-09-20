"""Application updater with official GitHub Releases, staged SHA256/zip validation, rollback, and restart."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import application_root, default_data_directory
from .version import GITHUB_REPO, __version__, is_newer_version, parse_version


class UpdateSecurityError(RuntimeError):
    pass


class UpdateValidationError(RuntimeError):
    pass


@dataclass
class ReleaseAsset:
    name: str
    url: str
    size: int = 0
    content_type: str = ""


@dataclass
class ReleaseInfo:
    tag_name: str
    version: str
    name: str
    body: str
    html_url: str
    published_at: str
    assets: list[ReleaseAsset] = field(default_factory=list)
    selected_asset: ReleaseAsset | None = None
    sha256_asset: ReleaseAsset | None = None


def select_portable_asset(assets: list[ReleaseAsset]) -> ReleaseAsset | None:
    """Select the best portable ZIP archive asset from release assets."""
    candidates: list[tuple[int, ReleaseAsset]] = []
    for asset in assets:
        name = asset.name.lower()
        if not name.endswith(".zip"):
            continue
        score = 10
        if "portable" in name:
            score += 50
        if "toolrecap" in name or "recap" in name:
            score += 30
        if "win" in name or "windows" in name or "x64" in name:
            score += 20
        if "setup" in name or "installer" in name:
            score -= 100
        if score > 0:
            candidates.append((score, asset))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def parse_release_payload(payload: dict) -> ReleaseInfo:
    """Parse GitHub release JSON payload."""
    tag = str(payload.get("tag_name", "")).strip()
    version = tag.lstrip("vV")
    name = str(payload.get("name") or tag or "Bản phát hành mới")
    body = str(payload.get("body") or "").strip()
    html_url = str(payload.get("html_url") or "")
    published_at = str(payload.get("published_at") or "")

    assets: list[ReleaseAsset] = []
    sha256_asset: ReleaseAsset | None = None

    for raw in payload.get("assets", []):
        if isinstance(raw, dict):
            asset_name = str(raw.get("name", ""))
            download_url = str(raw.get("browser_download_url", ""))
            size = int(raw.get("size", 0))
            content_type = str(raw.get("content_type", ""))
            if asset_name and download_url:
                asset = ReleaseAsset(
                    name=asset_name,
                    url=download_url,
                    size=size,
                    content_type=content_type,
                )
                assets.append(asset)
                if asset_name.lower().endswith((".sha256", ".sha256.txt")):
                    sha256_asset = asset

    selected = select_portable_asset(assets)
    return ReleaseInfo(
        tag_name=tag,
        version=version,
        name=name,
        body=body,
        html_url=html_url,
        published_at=published_at,
        assets=assets,
        selected_asset=selected,
        sha256_asset=sha256_asset,
    )


def fetch_latest_release(repo: str = GITHUB_REPO, timeout: float = 10.0) -> ReleaseInfo | None:
    """Fetch latest public release from GitHub without requiring a token."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"ToolRecapV2/{__version__} (Windows)",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))
            return parse_release_payload(data)
    except Exception:
        return None


def check_for_updates(
    current_version: str = __version__,
    repo: str = GITHUB_REPO,
    timeout: float = 10.0,
) -> ReleaseInfo | None:
    """Check if a strictly newer release is available with a portable asset."""
    release = fetch_latest_release(repo=repo, timeout=timeout)
    if not release or not release.selected_asset:
        return None

    if is_newer_version(current_version, release.version):
        return release
    return None


def verify_zip_safety(zip_file: Path | zipfile.ZipFile) -> None:
    """Ensure zip archive does not contain path traversal (zip-slip) or absolute paths."""
    zf = zip_file if isinstance(zip_file, zipfile.ZipFile) else zipfile.ZipFile(zip_file)
    try:
        for name in zf.namelist():
            # Check for absolute paths
            if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
                raise UpdateSecurityError(f"Phát hiện đường dẫn tuyệt đối nguy hiểm trong file ZIP: {name}")

            # Check for path traversal ..
            parts = Path(name).parts
            if ".." in parts:
                raise UpdateSecurityError(f"Phát hiện nguy cơ zip-slip (..) trong file ZIP: {name}")
    finally:
        if not isinstance(zip_file, zipfile.ZipFile):
            zf.close()


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract zip archive safely after verifying all paths."""
    verify_zip_safety(zip_path)
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target_dir / member.filename).resolve()
            # Double check that destination stays inside target_dir
            if not str(dest).startswith(str(target_dir)):
                raise UpdateSecurityError(f"Đích giải nén nằm ngoài thư mục mục tiêu: {member.filename}")
            zf.extract(member, target_dir)


def extract_expected_sha256(sha_content: str, target_filename: str) -> str:
    """Extract SHA256 hex digest for target_filename from sha256 checksum file content.
    Bare hash is accepted only when content is exactly one hash token.
    Named checksum line must match target_filename.
    """
    clean_content = sha_content.strip()
    if not clean_content:
        raise UpdateValidationError("Tệp SHA256 rỗng.")

    tokens = clean_content.split()
    # Bare hash accepted only when content is exactly one hash token
    if len(tokens) == 1:
        candidate = tokens[0].lower()
        if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate):
            return candidate

    # Otherwise, inspect lines for named checksum matching target_filename
    target_clean = target_filename.lower().strip()
    for line in clean_content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            fname = parts[-1].lstrip("*").lower()
            if fname == target_clean or Path(fname).name == target_clean:
                candidate_hash = parts[0].strip().lower()
                if len(candidate_hash) == 64 and all(c in "0123456789abcdef" for c in candidate_hash):
                    return candidate_hash

    raise UpdateValidationError(
        f"Tệp SHA256 không chứa chữ ký khớp với tệp '{target_filename}'."
    )


def download_and_stage_update(
    release: ReleaseInfo,
    *,
    staging_dir: Path | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Download the release zip, verify required SHA256, and safely extract to staging directory."""
    if not release.selected_asset:
        raise UpdateValidationError("Không tìm thấy asset zip portable trong bản phát hành.")

    # 1. Release SHA256 is REQUIRED. Fail closed if missing.
    if not release.sha256_asset:
        raise UpdateValidationError(
            "Bản phát hành thiếu tệp mã băm SHA256 bắt buộc. Từ chối cập nhật để đảm bảo an toàn."
        )

    staging = staging_dir or (default_data_directory() / "staging" / f"update_{release.version}")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    zip_dest = staging / release.selected_asset.name

    try:
        # Download ZIP
        req = urllib.request.Request(
            release.selected_asset.url,
            headers={"User-Agent": f"ToolRecapV2/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_dest, "wb") as out:
            total = int(resp.headers.get("Content-Length") or release.selected_asset.size or 0)
            downloaded = 0
            while True:
                if cancel_event and cancel_event.is_set():
                    raise UpdateValidationError("Tải bản cập nhật đã bị hủy.")
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(downloaded, total, min(100.0, (downloaded / total) * 100.0))

        # 2. Check SHA256. Fail closed if unfetchable or mismatch.
        sha_req = urllib.request.Request(
            release.sha256_asset.url,
            headers={"User-Agent": f"ToolRecapV2/{__version__}"},
        )
        try:
            with urllib.request.urlopen(sha_req, timeout=15) as s_resp:
                sha_content = s_resp.read().decode("utf-8").strip()
        except Exception as exc:
            raise UpdateValidationError(
                f"Không thể tải tệp xác thực SHA256 từ GitHub ({exc}). Từ chối cập nhật để đảm bảo an toàn."
            ) from exc

        expected_sha = extract_expected_sha256(sha_content, release.selected_asset.name)
        actual_sha = hashlib.sha256(zip_dest.read_bytes()).hexdigest().lower()
        if actual_sha != expected_sha:
            raise UpdateValidationError(
                f"Kiểm tra tính toàn vẹn SHA256 không khớp ({actual_sha} != {expected_sha})."
            )

        # 3. Safe extract
        extract_dir = staging / "extracted"
        safe_extract_zip(zip_dest, extract_dir)

        # If extracted content is wrapped in a single root folder, unwrap it
        sub_items = [p for p in extract_dir.iterdir()]
        if len(sub_items) == 1 and sub_items[0].is_dir():
            payload_dir = sub_items[0]
        else:
            payload_dir = extract_dir

        # Validate payload contains ToolRecapV2.exe
        exe_candidate = payload_dir / "ToolRecapV2.exe"
        if not exe_candidate.is_file():
            raise UpdateValidationError(
                "Gói cập nhật không chứa tệp thực thi ToolRecapV2.exe hợp lệ."
            )

        return payload_dir
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_apply_script(
    staged_payload_dir: Path,
    target_app_dir: Path,
    exe_name: str = "ToolRecapV2.exe",
) -> Path:
    """Generate a Windows batch script to swap files transactionally and restart the app.
    Waits for old process to exit, completely swaps directory (so stale files are not retained),
    preserves external %LOCALAPPDATA%, and rolls back on failure.
    """
    staged = Path(staged_payload_dir).resolve()
    target = Path(target_app_dir).resolve()
    script_path = default_data_directory() / "apply_update.cmd"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = default_data_directory() / "backups" / "previous_version"

    content = f"""@echo off
chcp 65001 >nul
echo [ToolRecap V2] Đang chuẩn bị cập nhật...

set "TARGET={str(target)}"
set "STAGED={str(staged)}"
set "BACKUP={str(backup_dir)}"
set "EXE_NAME={exe_name}"

echo [ToolRecap V2] Đang chờ ứng dụng cũ thoát hoàn toàn...
:wait_exit
tasklist /fi "imagename eq %EXE_NAME%" 2>nul | find /i "%EXE_NAME%" >nul
if not errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_exit
)
timeout /t 1 /nobreak >nul

echo [ToolRecap V2] Tạo bản sao lưu phiên bản hiện tại...
if exist "%BACKUP%" rmdir /s /q "%BACKUP%"
mkdir "%BACKUP%" 2>nul
xcopy "%TARGET%\\*" "%BACKUP%\\" /e /i /h /y >nul
if errorlevel 1 goto backup_failed
if not exist "%BACKUP%\\%EXE_NAME%" goto backup_failed

echo [ToolRecap V2] Xác minh bản sao lưu thành công.

echo [ToolRecap V2] Làm sạch thư mục ứng dụng (loại bỏ tệp cũ thừa)...
for /d %%p in ("%TARGET%\\*") do rmdir /s /q "%%p" 2>nul
for %%f in ("%TARGET%\\*") do del /f /q "%%f" 2>nul

echo [ToolRecap V2] Áp dụng dữ liệu phiên bản mới...
xcopy "%STAGED%\\*" "%TARGET%\\" /e /i /h /y >nul
if errorlevel 1 goto apply_failed
if not exist "%TARGET%\\%EXE_NAME%" goto apply_failed

echo [ToolRecap V2] Xác minh bản cài đặt mới thành công.

echo [ToolRecap V2] Dọn dẹp tệp tải về tạm thời...
rmdir /s /q "%STAGED%" 2>nul

echo [ToolRecap V2] Khởi động lại ToolRecap V2...
start "" "%TARGET%\\%EXE_NAME%"
exit 0

:backup_failed
echo [ToolRecap V2] Sao lưu không thành công. Bản cài đặt hiện tại chưa bị thay đổi.
pause
exit /b 2

:apply_failed
echo [ToolRecap V2] Áp dụng thất bại. Đang hoàn tác về bản sao lưu đã xác minh...
for /d %%p in ("%TARGET%\\*") do rmdir /s /q "%%p" 2>nul
for %%f in ("%TARGET%\\*") do del /f /q "%%f" 2>nul
xcopy "%BACKUP%\\*" "%TARGET%\\" /e /i /h /y >nul
if errorlevel 1 goto rollback_failed
if not exist "%TARGET%\\%EXE_NAME%" goto rollback_failed
echo [ToolRecap V2] Hoàn tác thành công. Khởi động lại phiên bản trước...
start "" "%TARGET%\\%EXE_NAME%"
exit /b 1

:rollback_failed
echo [ToolRecap V2] LỖI NGHIÊM TRỌNG: Không thể hoàn tác tự động. Bản sao lưu vẫn ở "%BACKUP%".
pause
exit /b 3
"""
    script_path.write_text(content, encoding="utf-8")
    return script_path

"""Voice subsystem updater with allowlisted sources, staged validation, and rollback safety."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import default_data_directory


ALLOWLISTED_VOICE_REPOS = frozenset({
    "debpalash/VoiceStudio",
    "longthao9820-alt/tool-recap-v2",
    "rhasspy/piper-voices",
})

OFFICIAL_VOICESTUDIO_REPO = "debpalash/VoiceStudio"
PINNED_VOICESTUDIO_VERSION = "0.5.3"
PINNED_VOICESTUDIO_TAG = "v0.5.3"
COMPATIBLE_VOICESTUDIO_TAGS = frozenset({"v0.5.3", "0.5.3"})
ALLOWED_ADAPTER_EXECUTABLES = frozenset({"VoiceStudio.exe", "adapter.exe", "voicestudio_adapter.exe"})


class VoiceUpdateSecurityError(RuntimeError):
    pass


class VoiceUpdateValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceManifestAsset:
    filename: str
    sha256: str
    size: int
    url: str
    digest: str = ""


@dataclass(frozen=True)
class VoiceStudioReleaseInfo:
    tag_name: str
    version: str
    html_url: str
    published_at: str
    is_compatible: bool
    is_newer_incompatible: bool
    assets: tuple[VoiceManifestAsset, ...] = ()
    download_url: str = ""


@dataclass(frozen=True)
class VoiceStudioStatus:
    installed_version: str | None
    supported_version: str = PINNED_VOICESTUDIO_VERSION
    latest_version: str | None = None
    is_compatible: bool = False
    is_newer_incompatible: bool = False
    has_adapter: bool = False
    adapter_asset: VoiceManifestAsset | None = None
    status_label: str = ""
    release_info: VoiceStudioReleaseInfo | None = None


def get_installed_voicestudio_version(subsystem_dir: Path | None = None) -> str | None:
    """Read currently installed VoiceStudio subsystem version."""
    target = subsystem_dir or (default_data_directory() / "voice_subsystem")
    if not target.is_dir():
        return None

    manifest_file = target / "voice_manifest.json"
    if manifest_file.is_file():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            ver = str(data.get("subsystem_version", "")).strip()
            if ver:
                return ver
        except Exception:
            pass

    info_file = target / "subsystem_info.json"
    if info_file.is_file():
        try:
            data = json.loads(info_file.read_text(encoding="utf-8"))
            ver = str(data.get("version", "")).strip()
            if ver:
                return ver
        except Exception:
            pass

    return None


def parse_voicestudio_release(payload: dict[str, Any]) -> VoiceStudioReleaseInfo:
    """Parse debpalash/VoiceStudio GitHub release payload and verify compatibility."""
    tag = str(payload.get("tag_name", "")).strip()
    version = tag.lstrip("vV")
    html_url = str(payload.get("html_url", ""))
    published_at = str(payload.get("published_at", ""))

    is_compatible = tag in COMPATIBLE_VOICESTUDIO_TAGS or version == PINNED_VOICESTUDIO_VERSION

    # Reject incompatible newer releases (e.g. tag > v0.5.3)
    is_newer_incompatible = False
    if not is_compatible:
        try:
            from ..version import is_newer_version
            if is_newer_version(PINNED_VOICESTUDIO_VERSION, version):
                is_newer_incompatible = True
        except Exception:
            if version > PINNED_VOICESTUDIO_VERSION:
                is_newer_incompatible = True

    raw_assets = payload.get("assets", [])
    companion_hashes: dict[str, str] = {}
    for item in raw_assets:
        if not isinstance(item, dict):
            continue
        c_name = str(item.get("name", "")).strip()
        c_hash = str(item.get("sha256") or item.get("digest") or item.get("checksum") or "").strip().lower()
        if c_hash.startswith("sha256:"):
            c_hash = c_hash[7:].strip()
        if len(c_hash) == 64 and all(c in "0123456789abcdef" for c in c_hash):
            companion_hashes[c_name] = c_hash

    assets: list[VoiceManifestAsset] = []
    download_url = ""
    for item in raw_assets:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("name", "")).strip()
        url = str(item.get("browser_download_url", "")).strip()
        size = int(item.get("size", 0))
        digest_val = str(item.get("digest") or "").strip()

        extracted_sha = ""
        if digest_val.lower().startswith("sha256:"):
            cand = digest_val.lower().split("sha256:", 1)[1].strip()
            if len(cand) == 64 and all(c in "0123456789abcdef" for c in cand):
                extracted_sha = cand
        elif len(digest_val) == 64 and all(c in "0123456789abcdef" for c in digest_val.lower()):
            extracted_sha = digest_val.lower()

        if not extracted_sha:
            item_sha = str(item.get("sha256") or "").strip().lower()
            if item_sha.startswith("sha256:"):
                item_sha = item_sha[7:].strip()
            if len(item_sha) == 64 and all(c in "0123456789abcdef" for c in item_sha):
                extracted_sha = item_sha

        # Check companion checksum asset (e.g. {fname}.sha256)
        if not extracted_sha:
            for comp in (f"{fname}.sha256", f"{fname}.sha256.txt"):
                if comp in companion_hashes:
                    extracted_sha = companion_hashes[comp]
                    break

        assets.append(
            VoiceManifestAsset(
                filename=fname,
                sha256=extracted_sha,
                size=size,
                url=url,
                digest=digest_val,
            )
        )
        if fname.lower().endswith(".zip") and not download_url:
            download_url = url

    return VoiceStudioReleaseInfo(
        tag_name=tag,
        version=version,
        html_url=html_url,
        published_at=published_at,
        is_compatible=is_compatible,
        is_newer_incompatible=is_newer_incompatible,
        assets=tuple(assets),
        download_url=download_url,
    )


def find_compatible_adapter_asset(release_info: VoiceStudioReleaseInfo) -> VoiceManifestAsset | None:
    """Find a V2-compatible adapter package asset from release assets.
    Must not select arbitrary full desktop application archives.
    Only packages explicitly matching adapter naming (e.g. *adapter*.zip) AND
    having a trusted SHA256 checksum (from GitHub release asset digest or matching companion checksum)
    are accepted. Unsigned / unverified packages are rejected.
    """
    for asset in release_info.assets:
        name_lower = asset.filename.lower()
        if name_lower.endswith(".zip") and "adapter" in name_lower:
            if asset.sha256 and len(asset.sha256) == 64 and all(c in "0123456789abcdef" for c in asset.sha256.lower()):
                return asset
    return None


def download_and_stage_voicestudio_adapter(
    asset: VoiceManifestAsset,
    *,
    staging_dir: Path | None = None,
    progress_callback: Any | None = None,
    cancel_event: Any | None = None,
) -> Path:
    """Download and safely stage a VoiceStudio adapter package, validating its manifest and executable adapter."""
    from ..updater import safe_extract_zip

    staging = staging_dir or (default_data_directory() / "staging" / "voicestudio_adapter")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    zip_dest = staging / asset.filename
    try:
        import urllib.request
        req = urllib.request.Request(
            asset.url,
            headers={"User-Agent": "ToolRecapV2/VoiceStudioUpdater"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_dest, "wb") as out:
            total = int(resp.headers.get("Content-Length") or asset.size or 0)
            downloaded = 0
            while True:
                if cancel_event and cancel_event.is_set():
                    raise VoiceUpdateValidationError("Tải adapter VoiceStudio đã bị hủy.")
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(downloaded, total, min(100.0, (downloaded / total) * 100.0))

        # 1. Require and validate trusted SHA256 before extraction
        if not asset.sha256 or len(asset.sha256) != 64:
            raise VoiceUpdateSecurityError("Gói adapter thiếu mã băm SHA256 / digest an toàn.")
        calc_sha = hashlib.sha256(zip_dest.read_bytes()).hexdigest().lower()
        if calc_sha != asset.sha256.lower():
            raise VoiceUpdateValidationError(
                f"Mã SHA256 của file tải về không khớp: thực tế {calc_sha} != yêu cầu {asset.sha256}"
            )

        # 2. Extract safely
        extract_dir = staging / "extracted"
        safe_extract_zip(zip_dest, extract_dir)

        sub_items = [p for p in extract_dir.iterdir()]
        payload_dir = sub_items[0] if len(sub_items) == 1 and sub_items[0].is_dir() else extract_dir

        # 3. Require manifest voice_manifest.json
        manifest_path = payload_dir / "voice_manifest.json"
        if not manifest_path.is_file():
            raise VoiceUpdateValidationError("Gói adapter bắt buộc phải chứa tệp 'voice_manifest.json'.")

        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise VoiceUpdateValidationError(f"Tệp 'voice_manifest.json' không đúng định dạng JSON: {exc}") from exc

        if not isinstance(m_data, dict):
            raise VoiceUpdateValidationError("Manifest phải là một đối tượng JSON.")

        # Validate repository exactly official debpalash/VoiceStudio
        m_repo = str(m_data.get("repository", "")).strip()
        if m_repo != OFFICIAL_VOICESTUDIO_REPO:
            raise VoiceUpdateSecurityError(
                f"Repository trong manifest '{m_repo}' không hợp lệ. Phải chính xác là '{OFFICIAL_VOICESTUDIO_REPO}'."
            )

        # Validate subsystem version compatible (v0.5.3)
        m_sub_ver = str(m_data.get("subsystem_version", "")).strip()
        m_sub_clean = m_sub_ver.lstrip("vV")
        if not (m_sub_ver in COMPATIBLE_VOICESTUDIO_TAGS or m_sub_clean == PINNED_VOICESTUDIO_VERSION):
            raise VoiceUpdateValidationError(
                f"Phiên bản subsystem '{m_sub_ver}' không tương thích (yêu cầu {PINNED_VOICESTUDIO_VERSION})."
            )

        # Validate min_app_version <= current app
        m_min_app = str(m_data.get("min_app_version", "")).strip()
        if m_min_app:
            from ..version import __version__, is_newer_version
            if is_newer_version(__version__, m_min_app):
                raise VoiceUpdateValidationError(
                    f"Gói adapter yêu cầu phiên bản ứng dụng tối thiểu {m_min_app}, hiện tại là {__version__}."
                )

        # Validate adapter executable filename/path safe and exists
        m_exe = str(m_data.get("adapter_executable") or m_data.get("executable") or "").strip()
        if m_exe:
            if "/" in m_exe or "\\" in m_exe or ".." in m_exe:
                raise VoiceUpdateSecurityError(f"Đường dẫn adapter executable không an toàn: {m_exe}")
            if m_exe not in ALLOWED_ADAPTER_EXECUTABLES:
                raise VoiceUpdateSecurityError(
                    f"Tệp thực thi adapter '{m_exe}' không nằm trong allowlist."
                )

        found_exes = [name for name in ALLOWED_ADAPTER_EXECUTABLES if (payload_dir / name).is_file()]
        if not found_exes:
            raise VoiceUpdateValidationError(
                "Gói adapter không chứa tệp thực thi adapter hợp lệ (VoiceStudio.exe / adapter.exe)."
            )

        # Reject any other executable / script files in payload_dir outside allowlist
        for f in payload_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".exe", ".bat", ".cmd", ".ps1", ".vbs"):
                if f.name not in ALLOWED_ADAPTER_EXECUTABLES:
                    raise VoiceUpdateSecurityError(
                        f"Phát hiện tệp thực thi không được phép trong gói adapter: {f.name}"
                    )

        return payload_dir
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def check_voicestudio_status(
    subsystem_dir: Path | None = None,
    repo: str = OFFICIAL_VOICESTUDIO_REPO,
    timeout: float = 10.0,
    *,
    mock_payload: dict[str, Any] | None = None,
) -> VoiceStudioStatus:
    """Check VoiceStudio subsystem: installed, supported, and latest release compatibility."""
    installed = get_installed_voicestudio_version(subsystem_dir=subsystem_dir)

    payload = mock_payload
    if payload is None:
        try:
            import urllib.request
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "ToolRecapV2/VoiceStudioChecker",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass

    if not payload:
        status_lbl = (
            f"Đã cài đặt: {installed or 'Chưa cài đặt'} · Hỗ trợ: {PINNED_VOICESTUDIO_VERSION} (Ngoại tuyến)"
        )
        return VoiceStudioStatus(
            installed_version=installed,
            supported_version=PINNED_VOICESTUDIO_VERSION,
            latest_version=None,
            is_compatible=installed == PINNED_VOICESTUDIO_VERSION,
            is_newer_incompatible=False,
            has_adapter=False,
            adapter_asset=None,
            status_label=status_lbl,
            release_info=None,
        )

    rel_info = parse_voicestudio_release(payload)
    adapter_asset = find_compatible_adapter_asset(rel_info)
    has_adapter = adapter_asset is not None

    has_unsigned_adapter = any(
        a.filename.lower().endswith(".zip") and "adapter" in a.filename.lower()
        for a in rel_info.assets
    ) and not has_adapter

    if rel_info.is_newer_incompatible:
        status_lbl = (
            f"Bản phát hành mới nhất ({rel_info.tag_name}) chưa kiểm định tương thích. "
            f"Giữ nguyên phiên bản hiện tại ({installed or PINNED_VOICESTUDIO_VERSION})."
        )
    elif not rel_info.is_compatible:
        status_lbl = f"Bản phát hành {rel_info.tag_name} không tương thích với ToolRecap V2."
    elif has_unsigned_adapter:
        status_lbl = (
            f"Bản phát hành {rel_info.tag_name} có gói adapter nhưng thiếu mã băm SHA256 / digest an toàn (bị từ chối). "
            f"Giữ nguyên hệ thống hiện tại."
        )
    elif not has_adapter:
        status_lbl = (
            f"Bản phát hành {rel_info.tag_name} là ứng dụng desktop độc lập, chưa có gói adapter "
            f"tương thích với ToolRecap V2. Giữ nguyên hệ thống hiện tại."
        )
    elif installed == rel_info.version:
        status_lbl = f"VoiceStudio đang ở phiên bản tương thích ({installed})."
    else:
        status_lbl = f"Có gói adapter VoiceStudio tương thích ({rel_info.tag_name}) sẵn sàng cài đặt."

    return VoiceStudioStatus(
        installed_version=installed,
        supported_version=PINNED_VOICESTUDIO_VERSION,
        latest_version=rel_info.version,
        is_compatible=rel_info.is_compatible,
        is_newer_incompatible=rel_info.is_newer_incompatible,
        has_adapter=has_adapter,
        adapter_asset=adapter_asset,
        status_label=status_lbl,
        release_info=rel_info,
    )


def apply_voicestudio_subsystem_update(
    staged_dir: Path,
    target_dir: Path | None = None,
    version: str = PINNED_VOICESTUDIO_VERSION,
) -> Path:
    """Transactionally clean-swap VoiceStudio subsystem update while preserving model cache."""
    target = target_dir or (default_data_directory() / "voice_subsystem")
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    backup_dir = default_data_directory() / "backups" / f"voicestudio_backup_{version}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    try:
        # 1. Backup existing files in target
        if any(target.iterdir()):
            shutil.copytree(target, backup_dir)

        # 2. Clean swap: remove stale files from target
        for item in list(target.iterdir()):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

        # 3. Copy staged files into target
        for item in staged_dir.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # 4. Write subsystem info metadata
        info_file = target / "subsystem_info.json"
        info_file.write_text(
            json.dumps({"version": version, "repository": OFFICIAL_VOICESTUDIO_REPO}, indent=2),
            encoding="utf-8",
        )
        return target
    except Exception as exc:
        # Clean rollback on failure
        for item in list(target.iterdir()):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        if backup_dir.exists():
            for item in backup_dir.iterdir():
                dest = target / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        raise VoiceUpdateValidationError(f"Cập nhật VoiceStudio thất bại, đã rollback: {exc}") from exc
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)


@dataclass(frozen=True)
class VoicePackageManifest:
    subsystem_version: str
    min_app_version: str
    repository: str
    voices: tuple[str, ...]
    assets: tuple[VoiceManifestAsset, ...]


def parse_and_validate_voice_manifest(raw_json: str | dict[str, Any]) -> VoicePackageManifest:
    """Parse and validate a voice package manifest against security and schema rules."""
    if isinstance(raw_json, str):
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise VoiceUpdateValidationError(f"JSON manifest không hợp lệ: {exc}") from exc
    else:
        data = raw_json

    if not isinstance(data, dict):
        raise VoiceUpdateValidationError("Manifest phải là một JSON object.")

    repo = str(data.get("repository", "")).strip()
    if repo not in ALLOWLISTED_VOICE_REPOS:
        raise VoiceUpdateSecurityError(
            f"Nguồn cập nhật '{repo}' không nằm trong danh sách được phép (allowlist)."
        )

    subsystem_ver = str(data.get("subsystem_version", "")).strip()
    min_app_ver = str(data.get("min_app_version", "")).strip()
    if not subsystem_ver:
        raise VoiceUpdateValidationError("Manifest thiếu trường 'subsystem_version'.")

    raw_voices = data.get("voices", [])
    if not isinstance(raw_voices, list) or not raw_voices:
        raise VoiceUpdateValidationError("Manifest phải chứa danh sách giọng nói ('voices').")

    raw_assets = data.get("assets", [])
    if not isinstance(raw_assets, list):
        raise VoiceUpdateValidationError("Manifest thiếu trường 'assets'.")

    assets: list[VoiceManifestAsset] = []
    for item in raw_assets:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("filename", "")).strip()
        sha = str(item.get("sha256", "")).strip().lower()
        size = int(item.get("size", 0))
        url = str(item.get("url", "")).strip()

        # Reject path traversal
        if ".." in fname or fname.startswith(("/", "\\")):
            raise VoiceUpdateSecurityError(f"Tên tệp không an toàn trong manifest: {fname}")

        assets.append(VoiceManifestAsset(filename=fname, sha256=sha, size=size, url=url))

    return VoicePackageManifest(
        subsystem_version=subsystem_ver,
        min_app_version=min_app_ver,
        repository=repo,
        voices=tuple(str(v) for v in raw_voices),
        assets=tuple(assets),
    )


def validate_staged_voice_assets(
    staged_dir: Path,
    manifest: VoicePackageManifest,
) -> None:
    """Validate that all files in staged_dir match the manifest hashes and sizes."""
    for asset in manifest.assets:
        file_path = staged_dir / asset.filename
        if not file_path.is_file():
            raise VoiceUpdateValidationError(f"Thiếu file sau khi tải: {asset.filename}")

        if asset.size > 0 and file_path.stat().st_size != asset.size:
            raise VoiceUpdateValidationError(
                f"Kích thước file {asset.filename} không khớp: "
                f"thực tế {file_path.stat().st_size} != dự kiến {asset.size}"
            )

        if asset.sha256:
            calc_sha = hashlib.sha256(file_path.read_bytes()).hexdigest().lower()
            if calc_sha != asset.sha256:
                raise VoiceUpdateValidationError(
                    f"Mã SHA256 file {asset.filename} không khớp: "
                    f"{calc_sha} != {asset.sha256}"
                )


def apply_staged_voice_update(
    staged_dir: Path,
    target_dir: Path,
    manifest: VoicePackageManifest,
) -> None:
    """Transactionally apply staged voice update to target directory with rollback safety.
    Preserves existing models and existing subsystem if anything fails.
    """
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = default_data_directory() / "backups" / f"voice_subsystem_backup_{manifest.subsystem_version}"

    # 1. Validate staged assets first
    validate_staged_voice_assets(staged_dir, manifest)

    # 2. Stage backup of existing files that will be overwritten
    backed_up_files: list[Path] = []
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for asset in manifest.assets:
            dest_file = target_dir / asset.filename
            if dest_file.is_file():
                bak_file = backup_dir / asset.filename
                bak_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest_file, bak_file)
                backed_up_files.append(dest_file)

        # 3. Copy staged files to target
        for asset in manifest.assets:
            src_file = staged_dir / asset.filename
            dest_file = target_dir / asset.filename
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)

        # 4. Write new manifest
        manifest_dest = target_dir / "voice_manifest.json"
        manifest_dest.write_text(
            json.dumps({
                "subsystem_version": manifest.subsystem_version,
                "repository": manifest.repository,
                "voices": list(manifest.voices),
                "assets": [
                    {"filename": a.filename, "sha256": a.sha256, "size": a.size, "url": a.url}
                    for a in manifest.assets
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        # Rollback on any failure
        for bak_dest in backed_up_files:
            rel = bak_dest.relative_to(target_dir)
            orig_backup = backup_dir / rel
            if orig_backup.is_file():
                shutil.copy2(orig_backup, bak_dest)
        raise VoiceUpdateValidationError(f"Cập nhật giọng nói thất bại, đã rollback an toàn: {exc}") from exc
    finally:
        # Clean up staging directory
        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)

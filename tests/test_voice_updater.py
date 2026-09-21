"""Tests for voice subsystem updater, allowlisting, staged validation, and rollback safety."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from toolrecap_v2.voice.voice_updater import (
    ALLOWLISTED_VOICE_REPOS,
    VoicePackageManifest,
    VoiceUpdateSecurityError,
    VoiceUpdateValidationError,
    apply_staged_voice_update,
    parse_and_validate_voice_manifest,
    validate_staged_voice_assets,
)


def test_allowlist_enforcement() -> None:
    # Allowlisted repo
    valid_payload = {
        "repository": "longthao9820-alt/tool-recap-v2",
        "subsystem_version": "1.0.0",
        "min_app_version": "0.1.0",
        "voices": ["piper.en_US-lessac-medium"],
        "assets": [
            {
                "filename": "model.onnx",
                "sha256": "abc",
                "size": 100,
                "url": "https://example.com/model.onnx",
            }
        ],
    }
    manifest = parse_and_validate_voice_manifest(valid_payload)
    assert manifest.subsystem_version == "1.0.0"

    # Untrusted repo -> must raise VoiceUpdateSecurityError
    untrusted_payload = dict(valid_payload)
    untrusted_payload["repository"] = "malicious-user/evil-voice"
    with pytest.raises(VoiceUpdateSecurityError, match="không nằm trong danh sách được phép"):
        parse_and_validate_voice_manifest(untrusted_payload)


def test_manifest_path_traversal_rejection() -> None:
    evil_payload = {
        "repository": "longthao9820-alt/tool-recap-v2",
        "subsystem_version": "1.0.0",
        "min_app_version": "0.1.0",
        "voices": ["piper.en_US-lessac-medium"],
        "assets": [
            {
                "filename": "../../windows/system32/cmd.exe",
                "sha256": "abc",
                "size": 100,
                "url": "https://example.com/evil",
            }
        ],
    }
    with pytest.raises(VoiceUpdateSecurityError, match="Tên tệp không an toàn"):
        parse_and_validate_voice_manifest(evil_payload)


def test_staged_validation_hash_mismatch(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    asset_file = staged / "voice.onnx"
    asset_file.write_bytes(b"actual_data")

    manifest = parse_and_validate_voice_manifest({
        "repository": "longthao9820-alt/tool-recap-v2",
        "subsystem_version": "1.0.0",
        "min_app_version": "0.1.0",
        "voices": ["v1"],
        "assets": [
            {
                "filename": "voice.onnx",
                "sha256": "wrong_hash_12345",
                "size": len(b"actual_data"),
                "url": "https://example.com/voice.onnx",
            }
        ],
    })

    with pytest.raises(VoiceUpdateValidationError, match="SHA256 file voice.onnx không khớp"):
        validate_staged_voice_assets(staged, manifest)


def test_apply_staged_voice_update_and_rollback(tmp_path: Path) -> None:
    target = tmp_path / "active_voices"
    target.mkdir()
    orig_file = target / "voice.onnx"
    orig_file.write_bytes(b"original_voice_data")

    staged = tmp_path / "staged"
    staged.mkdir()
    new_data = b"new_updated_voice_data"
    new_file = staged / "voice.onnx"
    new_file.write_bytes(new_data)
    new_sha = hashlib.sha256(new_data).hexdigest()

    manifest = parse_and_validate_voice_manifest({
        "repository": "longthao9820-alt/tool-recap-v2",
        "subsystem_version": "2.0.0",
        "min_app_version": "0.1.0",
        "voices": ["v2"],
        "assets": [
            {
                "filename": "voice.onnx",
                "sha256": new_sha,
                "size": len(new_data),
                "url": "https://example.com/voice.onnx",
            }
        ],
    })

    # Apply successfully
    apply_staged_voice_update(staged, target, manifest)
    assert orig_file.read_bytes() == new_data
    assert (target / "voice_manifest.json").is_file()


def test_voicestudio_compatibility_and_rejection() -> None:
    from toolrecap_v2.voice.voice_updater import (
        OFFICIAL_VOICESTUDIO_REPO,
        PINNED_VOICESTUDIO_VERSION,
        check_voicestudio_status,
        parse_voicestudio_release,
    )

    assert OFFICIAL_VOICESTUDIO_REPO in ALLOWLISTED_VOICE_REPOS

    # 1. Pinned supported version v0.5.3
    compat_payload = {
        "tag_name": "v0.5.3",
        "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.3",
        "published_at": "2024-01-01T00:00:00Z",
        "assets": [{"name": "VoiceStudio-v0.5.3.zip", "browser_download_url": "https://example.com/vs.zip", "size": 1000}],
    }
    rel = parse_voicestudio_release(compat_payload)
    assert rel.is_compatible is True
    assert rel.is_newer_incompatible is False
    assert rel.version == PINNED_VOICESTUDIO_VERSION

    # 2. Incompatible newer version v0.6.0 -> Must be rejected, retaining current
    newer_payload = {
        "tag_name": "v0.6.0",
        "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.6.0",
        "published_at": "2024-05-01T00:00:00Z",
        "assets": [{"name": "VoiceStudio-v0.6.0.zip", "browser_download_url": "https://example.com/vs6.zip", "size": 1000}],
    }
    newer_rel = parse_voicestudio_release(newer_payload)
    assert newer_rel.is_compatible is False
    assert newer_rel.is_newer_incompatible is True

    # 3. Status check with mock
    status = check_voicestudio_status(mock_payload=newer_payload)
    assert status.is_newer_incompatible is True
    assert "chưa kiểm định tương thích" in status.status_label


def test_voicestudio_transactional_apply_preserves_models(tmp_path: Path) -> None:
    from toolrecap_v2.voice.voice_updater import apply_voicestudio_subsystem_update, VoiceUpdateValidationError

    subsystem_dir = tmp_path / "voice_subsystem"
    subsystem_dir.mkdir()
    (subsystem_dir / "old_adapter.py").write_text("old", encoding="utf-8")

    # Isolated model cache
    models_dir = tmp_path / "models" / "voices"
    models_dir.mkdir(parents=True)
    voice_file = models_dir / "piper.en_US-lessac-medium.onnx"
    voice_file.write_bytes(b"model_weights_unaffected")

    staged = tmp_path / "staged_vs"
    staged.mkdir()
    (staged / "VoiceStudio.exe").write_bytes(b"binary_code")

    # 1. Apply: clean swap removes stale files and preserves external model cache
    apply_voicestudio_subsystem_update(staged, target_dir=subsystem_dir, version="0.5.3")

    # Verify subsystem updated
    assert (subsystem_dir / "VoiceStudio.exe").is_file()
    assert (subsystem_dir / "subsystem_info.json").is_file()
    # Stale file must be cleaned out
    assert not (subsystem_dir / "old_adapter.py").exists()
    # Verify model cache is 100% preserved
    assert voice_file.is_file()
    assert voice_file.read_bytes() == b"model_weights_unaffected"

    # 2. Rollback test: simulate error during apply
    staged_bad = tmp_path / "staged_bad"
    staged_bad.mkdir()
    (staged_bad / "VoiceStudio.exe").write_bytes(b"new_code")

    # Re-create a known file before failing apply
    (subsystem_dir / "important_subsystem.dll").write_text("dll_content", encoding="utf-8")

    import shutil
    orig_copy2 = shutil.copy2
    call_count = 0

    def _failing_copy(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("Simulated disk error during copy")
        return orig_copy2(*args, **kwargs)

    shutil.copy2 = _failing_copy
    try:
        with pytest.raises(VoiceUpdateValidationError, match="rollback"):
            apply_voicestudio_subsystem_update(staged_bad, target_dir=subsystem_dir, version="0.5.3")
    finally:
        shutil.copy2 = orig_copy2

    # Verify rollback restored the original state
    assert (subsystem_dir / "important_subsystem.dll").is_file()
    assert (subsystem_dir / "important_subsystem.dll").read_text(encoding="utf-8") == "dll_content"


def test_dynamic_catalog_requires_executable_adapter(tmp_path: Path) -> None:
    from toolrecap_v2.voice.catalog import BUILTIN_VOICES, get_available_voices

    manifest_file = tmp_path / "voice_manifest.json"
    manifest_data = {
        "subsystem_version": "0.5.3",
        "installed_voices": [
            {
                "voice_id": "custom.narrator-deep",
                "display_name": "Deep Narrator",
                "engine": "piper",
                "files": ["deep.onnx"],
                "required_files": ["deep.onnx"],
            },
            {
                "voice_id": "omnivoice.fake-voice",
                "display_name": "Fake OmniVoice",
                "engine": "omnivoice",
                "files": ["fake.onnx"],
                "required_files": ["fake.onnx"],
            }
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Without executable adapter -> should NOT expose extra voices
    voices_no_adapter = get_available_voices(manifest_path=manifest_file)
    assert "custom.narrator-deep" not in voices_no_adapter
    assert "omnivoice.fake-voice" not in voices_no_adapter

    # External adapter manifests no longer participate in the production catalog.
    (tmp_path / "VoiceStudio.exe").write_bytes(b"MZ_EXE")
    voices_with_adapter = get_available_voices(manifest_path=manifest_file)
    assert "custom.narrator-deep" not in voices_with_adapter
    assert "omnivoice.fake-voice" not in voices_with_adapter


def test_adapter_asset_detection_and_validation(tmp_path: Path) -> None:
    import zipfile
    from toolrecap_v2.voice.voice_updater import (
        VoiceManifestAsset,
        VoiceStudioReleaseInfo,
        VoiceUpdateSecurityError,
        VoiceUpdateValidationError,
        check_voicestudio_status,
        download_and_stage_voicestudio_adapter,
        find_compatible_adapter_asset,
        parse_voicestudio_release,
    )

    # 1. Desktop full archive only -> find_compatible_adapter_asset returns None
    desktop_only_payload = {
        "tag_name": "v0.5.3",
        "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.3",
        "published_at": "2024-01-01T00:00:00Z",
        "assets": [
            {"name": "VoiceStudio-v0.5.3-win64.zip", "browser_download_url": "https://example.com/desktop.zip", "size": 50000000},
        ],
    }
    rel_desktop = parse_voicestudio_release(desktop_only_payload)
    assert find_compatible_adapter_asset(rel_desktop) is None

    # Status check indicates desktop archive without compatible adapter
    status = check_voicestudio_status(mock_payload=desktop_only_payload)
    assert status.has_adapter is False
    assert "chưa có gói adapter" in status.status_label

    # 2. Release with unsigned adapter asset -> must be rejected
    unsigned_payload = {
        "tag_name": "v0.5.3",
        "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.3",
        "published_at": "2024-01-01T00:00:00Z",
        "assets": [
            {"name": "voicestudio-adapter-v0.5.3.zip", "browser_download_url": "https://example.com/adapter.zip", "size": 1000000},
        ],
    }
    rel_unsigned = parse_voicestudio_release(unsigned_payload)
    assert find_compatible_adapter_asset(rel_unsigned) is None
    status_unsigned = check_voicestudio_status(mock_payload=unsigned_payload)
    assert status_unsigned.has_adapter is False
    assert "thiếu mã băm SHA256" in status_unsigned.status_label

    # 3. Release with signed/digest adapter asset -> recognized and accepted
    dummy_sha = "a" * 64
    signed_payload = {
        "tag_name": "v0.5.3",
        "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.3",
        "published_at": "2024-01-01T00:00:00Z",
        "assets": [
            {
                "name": "voicestudio-adapter-v0.5.3.zip",
                "browser_download_url": "https://example.com/adapter.zip",
                "size": 1000000,
                "digest": f"sha256:{dummy_sha}",
            },
        ],
    }
    rel_signed = parse_voicestudio_release(signed_payload)
    matched_asset = find_compatible_adapter_asset(rel_signed)
    assert matched_asset is not None
    assert matched_asset.filename == "voicestudio-adapter-v0.5.3.zip"
    assert matched_asset.sha256 == dummy_sha
    status_signed = check_voicestudio_status(mock_payload=signed_payload)
    assert status_signed.has_adapter is True

    # 4. Download and stage: hash mismatch before extraction
    zip_path = tmp_path / "test_pkg.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "data")
    actual_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    bad_hash_asset = VoiceManifestAsset(
        filename="test_pkg.zip",
        sha256="b" * 64,  # Wrong hash
        size=zip_path.stat().st_size,
        url=f"file:///{zip_path.as_posix()}",
    )
    with pytest.raises(VoiceUpdateValidationError, match="không khớp"):
        download_and_stage_voicestudio_adapter(bad_hash_asset, staging_dir=tmp_path / "stage_bad_hash")

    # 5. Missing voice_manifest.json
    good_hash_no_manifest_asset = VoiceManifestAsset(
        filename="test_pkg.zip",
        sha256=actual_sha,
        size=zip_path.stat().st_size,
        url=f"file:///{zip_path.as_posix()}",
    )
    with pytest.raises(VoiceUpdateValidationError, match="voice_manifest.json"):
        download_and_stage_voicestudio_adapter(good_hash_no_manifest_asset, staging_dir=tmp_path / "stage_no_manifest")

    # 6. Invalid repository in manifest -> must raise VoiceUpdateSecurityError
    zip_untrusted = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(zip_untrusted, "w") as zf:
        zf.writestr("voice_manifest.json", json.dumps({
            "repository": "attacker/VoiceStudio",
            "subsystem_version": "0.5.3",
        }))
        zf.writestr("VoiceStudio.exe", "fake_exe")
    untrusted_sha = hashlib.sha256(zip_untrusted.read_bytes()).hexdigest()
    untrusted_asset = VoiceManifestAsset(
        filename="untrusted.zip",
        sha256=untrusted_sha,
        size=zip_untrusted.stat().st_size,
        url=f"file:///{zip_untrusted.as_posix()}",
    )
    with pytest.raises(VoiceUpdateSecurityError, match="Repository"):
        download_and_stage_voicestudio_adapter(untrusted_asset, staging_dir=tmp_path / "stage_untrusted")

    # 7. Incompatible subsystem version in manifest
    zip_incompat_ver = tmp_path / "incompat_ver.zip"
    with zipfile.ZipFile(zip_incompat_ver, "w") as zf:
        zf.writestr("voice_manifest.json", json.dumps({
            "repository": "debpalash/VoiceStudio",
            "subsystem_version": "0.9.9",
        }))
        zf.writestr("VoiceStudio.exe", "fake_exe")
    incompat_sha = hashlib.sha256(zip_incompat_ver.read_bytes()).hexdigest()
    incompat_asset = VoiceManifestAsset(
        filename="incompat_ver.zip",
        sha256=incompat_sha,
        size=zip_incompat_ver.stat().st_size,
        url=f"file:///{zip_incompat_ver.as_posix()}",
    )
    with pytest.raises(VoiceUpdateValidationError, match="không tương thích"):
        download_and_stage_voicestudio_adapter(incompat_asset, staging_dir=tmp_path / "stage_incompat_ver")

    # 8. Incompatible min_app_version
    zip_incompat_app = tmp_path / "incompat_app.zip"
    with zipfile.ZipFile(zip_incompat_app, "w") as zf:
        zf.writestr("voice_manifest.json", json.dumps({
            "repository": "debpalash/VoiceStudio",
            "subsystem_version": "0.5.3",
            "min_app_version": "99.0.0",
        }))
        zf.writestr("VoiceStudio.exe", "fake_exe")
    incompat_app_sha = hashlib.sha256(zip_incompat_app.read_bytes()).hexdigest()
    incompat_app_asset = VoiceManifestAsset(
        filename="incompat_app.zip",
        sha256=incompat_app_sha,
        size=zip_incompat_app.stat().st_size,
        url=f"file:///{zip_incompat_app.as_posix()}",
    )
    with pytest.raises(VoiceUpdateValidationError, match="phiên bản ứng dụng tối thiểu"):
        download_and_stage_voicestudio_adapter(incompat_app_asset, staging_dir=tmp_path / "stage_incompat_app")

    # 9. Successful staging with valid manifest and executable
    zip_valid = tmp_path / "valid_adapter.zip"
    with zipfile.ZipFile(zip_valid, "w") as zf:
        zf.writestr("voice_manifest.json", json.dumps({
            "repository": "debpalash/VoiceStudio",
            "subsystem_version": "0.5.3",
            "min_app_version": "0.1.0",
            "adapter_executable": "VoiceStudio.exe",
        }))
        zf.writestr("VoiceStudio.exe", "real_adapter_binary")
    valid_sha = hashlib.sha256(zip_valid.read_bytes()).hexdigest()
    valid_asset = VoiceManifestAsset(
        filename="valid_adapter.zip",
        sha256=valid_sha,
        size=zip_valid.stat().st_size,
        url=f"file:///{zip_valid.as_posix()}",
    )
    payload_dir = download_and_stage_voicestudio_adapter(valid_asset, staging_dir=tmp_path / "stage_valid")
    assert (payload_dir / "VoiceStudio.exe").is_file()
    assert (payload_dir / "voice_manifest.json").is_file()

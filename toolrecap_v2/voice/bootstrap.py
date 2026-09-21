"""Manifest-driven isolated Python runtime bootstrap for ToolRecap V2 voice subsystem.

PowerShell-free, uses only urllib/zipfile/subprocess/hashlib.
Supports progress reporting, cancellation, hash verification, and atomic swap with rollback.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..paths import default_data_directory
from .runtime import (
    CORE_DEPENDENCIES,
    DEPENDENCY_MANIFEST_VERSION,
    VOICE_ENGINE_VERSION,
    VOICE_RUNTIME_SCHEMA,
    VOICE_RUNTIME_VERSION,
    runtime_fingerprint,
)

ProgressCallback = Callable[[int, int, float], None]


class BootstrapError(RuntimeError):
    def __init__(self, message: str, *, code: str = "VOICE_RUNTIME_ERROR") -> None:
        super().__init__(message)
        self.code = code


class BootstrapCancelled(BootstrapError):
    pass


class BootstrapValidationError(BootstrapError):
    pass


class BootstrapSecurityError(BootstrapError):
    pass


@dataclass(frozen=True)
class BootstrapAsset:
    name: str
    url: str
    sha256: str
    size: int = 0
    extract: bool = False


# Pinned official binaries for Windows amd64 isolated runtime
PINNED_PYTHON_ASSET = BootstrapAsset(
    name="python-3.11.9-embed-amd64.zip",
    url="https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
    sha256="009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b",
    size=11249023,
    extract=True,
)

PINNED_GET_PIP_ASSET = BootstrapAsset(
    name="get-pip.py",
    url="https://bootstrap.pypa.io/get-pip.py",
    sha256="fb24e693bab954209a063d90953621412ccad4a500905a726286e038f508ddf6",
    size=2230488,
    extract=False,
)

PINNED_PIP_PACKAGES: tuple[str, ...] = tuple(
    f"{name}=={version}" for name, version in CORE_DEPENDENCIES.items()
)

DEFAULT_PIP_PACKAGES: tuple[str, ...] = PINNED_PIP_PACKAGES

TRUSTED_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TRUSTED_PYPI_INDEX_URL = "https://pypi.org/simple"
DEFAULT_EXTRA_INDEX_URLS: tuple[str, ...] = (TRUSTED_PYTORCH_INDEX_URL,)


@dataclass(frozen=True)
class VoiceRuntimeBootstrapManifest:
    version: str = VOICE_RUNTIME_VERSION
    python_version: str = "3.11.9"
    schema_version: str = VOICE_RUNTIME_SCHEMA
    dependency_manifest_version: str = DEPENDENCY_MANIFEST_VERSION
    assets: tuple[BootstrapAsset, ...] = (PINNED_PYTHON_ASSET, PINNED_GET_PIP_ASSET)
    pip_packages: tuple[str, ...] = DEFAULT_PIP_PACKAGES
    extra_index_urls: tuple[str, ...] = DEFAULT_EXTRA_INDEX_URLS


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract zip archive while preventing directory traversal attacks."""
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not member_path.is_relative_to(target_dir):
                raise BootstrapSecurityError(
                    f"Phát hiện nguy cơ Path Traversal trong tệp nén: {member.filename}"
                )
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(member_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def configure_embed_pth(runtime_dir: Path) -> None:
    """Enable site-packages in embeddable Python by modifying the *._pth file."""
    for pth_file in runtime_dir.glob("*._pth"):
        lines = pth_file.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        has_site = False
        has_site_packages = False
        for line in lines:
            stripped = line.strip()
            if stripped == "#import site" or stripped == "import site":
                new_lines.append("import site")
                has_site = True
            else:
                new_lines.append(line)
            if stripped in ("Lib/site-packages", "site-packages"):
                has_site_packages = True

        if not has_site:
            new_lines.append("import site")
        if not has_site_packages:
            new_lines.append("Lib/site-packages")

        pth_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _classify_install_failure(stderr: str) -> tuple[str, str]:
    detail = str(stderr or "").strip()
    lowered = detail.casefold()
    if "resolutionimpossible" in lowered or "conflicting dependencies" in lowered:
        return (
            "DEPENDENCY_CONFLICT",
            "Voice runtime dependencies are incompatible. ToolRecap could not prepare the local voice engine.",
        )
    if "no matching distribution" in lowered or "could not find a version" in lowered:
        return "PACKAGE_NOT_FOUND", "A required voice runtime package is unavailable."
    if any(token in lowered for token in ("connection", "timed out", "temporary failure", "name resolution")):
        return "NETWORK_ERROR", "The voice runtime package download failed because of a network error."
    return "DEPENDENCY_INSTALL_ERROR", "Voice runtime dependency installation failed."


def _run_install_command(
    command: list[str],
    *,
    timeout: float,
    cancel_event: threading.Event | None,
) -> subprocess.CompletedProcess[str]:
    """Run an installer command with process-tree cancellation and hidden console."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if cancel_event is None:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
        )

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        creationflags=creationflags,
    )
    started = time.monotonic()
    try:
        while proc.poll() is None:
            if cancel_event.is_set():
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=10,
                            creationflags=creationflags,
                        )
                    else:
                        proc.terminate()
                finally:
                    raise BootstrapCancelled("Voice runtime installation was cancelled.")
            if time.monotonic() - started > timeout:
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=10,
                            creationflags=creationflags,
                        )
                    else:
                        proc.kill()
                finally:
                    raise BootstrapValidationError(
                        "Voice runtime dependency installation timed out.",
                        code="INSTALL_TIMEOUT",
                    )
            cancel_event.wait(0.1)
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(command, int(proc.returncode or 0), stdout, stderr)
    finally:
        if proc.poll() is None:
            proc.kill()


class VoiceRuntimeBootstrap:
    """Transactionally downloads and installs an isolated Python runtime with pinned dependencies."""

    def __init__(
        self,
        target_dir: Path | None = None,
        staging_dir: Path | None = None,
        manifest: VoiceRuntimeBootstrapManifest | None = None,
    ) -> None:
        self.target_dir = target_dir or (default_data_directory() / "voice_runtime" / "current")
        self.staging_dir = staging_dir or (default_data_directory() / "staging" / "voice_runtime")
        self.manifest = manifest or VoiceRuntimeBootstrapManifest()

    def is_installed(self) -> bool:
        """Check the managed runtime identity; package health is verified separately."""
        py_exe = self.target_dir / "python.exe"
        metadata = self.target_dir / "voice_runtime.json"
        if not py_exe.is_file() or not metadata.is_file():
            return False
        try:
            raw = json.loads(metadata.read_text(encoding="utf-8"))
            return (
                str(raw.get("voice_runtime_schema")) == VOICE_RUNTIME_SCHEMA
                and str(raw.get("voice_runtime_version")) == VOICE_RUNTIME_VERSION
                and str(raw.get("runtime_fingerprint")) == runtime_fingerprint()
            )
        except Exception:
            return False

    def download_asset(
        self,
        asset: BootstrapAsset,
        destination: Path,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        chunk_size: int = 65536,
    ) -> Path:
        """Download asset and verify its SHA256 checksum."""
        if cancel_event and cancel_event.is_set():
            raise BootstrapCancelled("Quá trình tải runtime đã bị hủy.")

        dest = Path(destination).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest.parent / f"{dest.name}.part"

        try:
            req = urllib.request.Request(asset.url, headers={"User-Agent": "ToolRecapV2/Bootstrap"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_dest, "wb") as out:
                total_size = int(resp.headers.get("Content-Length") or asset.size or 0)
                downloaded = 0
                hasher = hashlib.sha256()

                while True:
                    if cancel_event and cancel_event.is_set():
                        raise BootstrapCancelled("Quá trình tải runtime đã bị hủy.")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size, min(100.0, (downloaded / total_size) * 100.0))

            if cancel_event and cancel_event.is_set():
                raise BootstrapCancelled("Quá trình tải runtime đã bị hủy.")

            actual_sha = hasher.hexdigest().lower()
            if asset.sha256 and actual_sha != asset.sha256.lower():
                raise BootstrapValidationError(
                    f"Mã SHA256 của {asset.name} không khớp: thực tế {actual_sha} != yêu cầu {asset.sha256}"
                )

            os.replace(temp_dest, dest)
            return dest
        except Exception:
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except OSError:
                    pass
            raise

    def bootstrap(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        install_packages: bool = True,
        force: bool = False,
        validation_callback: Callable[[Path], None] | None = None,
    ) -> Path:
        """Execute staged bootstrap: download -> verify -> extract -> pip -> atomic swap."""
        if not force and self.is_installed():
            if progress_callback:
                progress_callback(100, 100, 100.0)
            return self.target_dir / "python.exe"

        # 1. Clean staging directory
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        extracted_runtime = self.staging_dir / "extracted_runtime"
        extracted_runtime.mkdir(parents=True, exist_ok=True)

        backup_dir = default_data_directory() / "backups" / f"runtime_backup_{self.manifest.version}"
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        try:
            # 2. Download and extract assets
            get_pip_path: Path | None = None
            for asset in self.manifest.assets:
                if cancel_event and cancel_event.is_set():
                    raise BootstrapCancelled("Cài đặt runtime đã bị hủy.")

                asset_dest = self.staging_dir / asset.name
                self.download_asset(
                    asset,
                    asset_dest,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )

                if asset.extract:
                    safe_extract_zip(asset_dest, extracted_runtime)
                elif asset.name == "get-pip.py":
                    get_pip_path = asset_dest

            # 3. Configure ._pth to enable site-packages
            configure_embed_pth(extracted_runtime)

            # 4. Bootstrap pip using get-pip.py
            py_exe = extracted_runtime / "python.exe"
            if not py_exe.is_file():
                raise BootstrapValidationError("Không tìm thấy python.exe sau khi giải nén runtime.")

            if get_pip_path and get_pip_path.is_file():
                if cancel_event and cancel_event.is_set():
                    raise BootstrapCancelled("Cài đặt runtime đã bị hủy.")

                cmd_pip = [str(py_exe), str(get_pip_path), "--no-warn-script-location"]
                proc_pip = _run_install_command(cmd_pip, timeout=120, cancel_event=cancel_event)
                if proc_pip.returncode != 0:
                    raise BootstrapValidationError(
                        f"Cài đặt pip thất bại (mã {proc_pip.returncode}): {proc_pip.stderr.strip()}",
                        code="PIP_BOOTSTRAP_ERROR",
                    )

            # 5. Install required pip packages if requested
            if install_packages and self.manifest.pip_packages:
                if cancel_event and cancel_event.is_set():
                    raise BootstrapCancelled("Cài đặt runtime đã bị hủy.")

                # Separate PyTorch packages from other packages so PyTorch CPU index
                # does not cause pip to seek transformers or other PyPI packages on the PyTorch index
                torch_packages = [
                    pkg for pkg in self.manifest.pip_packages
                    if any(pkg.startswith(prefix) for prefix in ("torch==", "torchaudio==", "torchvision=="))
                ]
                engine_packages = [
                    pkg for pkg in self.manifest.pip_packages
                    if pkg.startswith("omnivoice==")
                ]
                other_packages = [
                    pkg for pkg in self.manifest.pip_packages
                    if pkg not in torch_packages and pkg not in engine_packages
                ]

                # Invocation 1: PyTorch CPU wheels from PyTorch index
                if torch_packages:
                    pip_cmd_torch = [
                        str(py_exe),
                        "-m",
                        "pip",
                        "install",
                        "--no-warn-script-location",
                    ]
                    indexes = tuple(getattr(self.manifest, "extra_index_urls", ()))
                    if indexes:
                        pip_cmd_torch.extend(["--index-url", indexes[0]])
                        pip_cmd_torch.extend(["--extra-index-url", TRUSTED_PYPI_INDEX_URL])
                    cpu_torch_packages = [
                        pkg + "+cpu" if pkg in {"torch==2.8.0", "torchaudio==2.8.0"} else pkg
                        for pkg in torch_packages
                    ]
                    pip_cmd_torch.extend(cpu_torch_packages)

                    proc_torch = _run_install_command(pip_cmd_torch, timeout=1200, cancel_event=cancel_event)
                    if proc_torch.returncode != 0:
                        code, message = _classify_install_failure(proc_torch.stderr)
                        raise BootstrapValidationError(
                            f"[{code}] {message}\n{proc_torch.stderr.strip()}", code=code
                        )

                # Invocation 2: Other packages (transformers, accelerate, soundfile, omnivoice, etc.) from PyPI
                if other_packages:
                    if cancel_event and cancel_event.is_set():
                        raise BootstrapCancelled("Cài đặt runtime đã bị hủy.")

                    pip_cmd_other = [
                        str(py_exe),
                        "-m",
                        "pip",
                        "install",
                        "--no-warn-script-location",
                    ]
                    pip_cmd_other.extend(other_packages)

                    proc_other = _run_install_command(pip_cmd_other, timeout=1200, cancel_event=cancel_event)
                    if proc_other.returncode != 0:
                        code, message = _classify_install_failure(proc_other.stderr)
                        raise BootstrapValidationError(
                            f"[{code}] {message}\n{proc_other.stderr.strip()}", code=code
                        )

                # Install the public inference engine without its broad UI/training
                # dependency extras. Every inference dependency is pinned above.
                if engine_packages:
                    pip_cmd_engine = [
                        str(py_exe), "-m", "pip", "install", "--no-warn-script-location", "--no-deps",
                        *engine_packages,
                    ]
                    proc_engine = _run_install_command(pip_cmd_engine, timeout=600, cancel_event=cancel_event)
                    if proc_engine.returncode != 0:
                        code, message = _classify_install_failure(proc_engine.stderr)
                        raise BootstrapValidationError(
                            f"[{code}] {message}\n{proc_engine.stderr.strip()}", code=code
                        )

            # 6. Write authoritative metadata and validate the complete staged runtime.
            info_file = extracted_runtime / "voice_runtime.json"
            info_file.write_text(
                json.dumps(
                    {
                        "voice_runtime_schema": self.manifest.schema_version,
                        "voice_runtime_version": self.manifest.version,
                        "python_version": self.manifest.python_version,
                        "engine": "omnivoice",
                        "engine_version": VOICE_ENGINE_VERSION,
                        "dependency_manifest_version": self.manifest.dependency_manifest_version,
                        "runtime_fingerprint": runtime_fingerprint(),
                        "installed_packages": list(self.manifest.pip_packages),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if validation_callback is not None:
                validation_callback(py_exe)

            # 7. Atomically promote only the already-validated staged runtime.
            self.target_dir.parent.mkdir(parents=True, exist_ok=True)
            had_previous = self.target_dir.exists()
            if had_previous:
                os.replace(self.target_dir, backup_dir)
            try:
                os.replace(extracted_runtime, self.target_dir)
            except Exception:
                if self.target_dir.exists():
                    shutil.rmtree(self.target_dir, ignore_errors=True)
                if backup_dir.exists():
                    os.replace(backup_dir, self.target_dir)
                raise

            if progress_callback:
                progress_callback(100, 100, 100.0)

            return self.target_dir / "python.exe"

        except Exception as exc:
            # Clean rollback on any failure
            if backup_dir.exists() and not self.target_dir.exists():
                os.replace(backup_dir, self.target_dir)
            raise
        finally:
            if self.staging_dir.exists():
                shutil.rmtree(self.staging_dir, ignore_errors=True)
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

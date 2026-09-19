# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

repo = Path(SPECPATH)

piper_datas = collect_data_files("piper")
onnx_datas = collect_data_files("onnxruntime")
onnx_binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    [str(repo / "auto_main.py")],
    pathex=[str(repo)],
    binaries=onnx_binaries,
    datas=[
        (str(repo / "assets"), "assets"),
    ]
    + piper_datas
    + onnx_datas,
    hiddenimports=[
        "PIL",
        "PIL.Image",
        "soundfile",
        "onnxruntime",
        "faster_whisper",
        "piper",
        "piper.voice",
        "piper.config",
        "toolrecap_v2",
        "toolrecap_v2.version",
        "toolrecap_v2.paths",
        "toolrecap_v2.scanner",
        "toolrecap_v2.gpu",
        "toolrecap_v2.media",
        "toolrecap_v2.narration",
        "toolrecap_v2.settings",
        "toolrecap_v2.updater",
        "toolrecap_v2.notifications",
        "toolrecap_v2.projects",
        "toolrecap_v2.ui",
        "toolrecap_v2.voice",
        "toolrecap_v2.voice.catalog",
        "toolrecap_v2.voice.manager",
        "toolrecap_v2.voice.audio_preview",
        "toolrecap_v2.voice.voice_updater",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ToolRecapV2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(repo / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ToolRecapV2",
)

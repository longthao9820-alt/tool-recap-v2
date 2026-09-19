# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

repo = Path(SPECPATH)

model_exts = (".onnx", ".bin", ".pt", ".safetensors", ".model", ".tflite")
piper_datas = [d for d in collect_data_files("piper") if not any(str(d[0]).lower().endswith(ext) for ext in model_exts)]
onnx_datas = [d for d in collect_data_files("onnxruntime") if not any(str(d[0]).lower().endswith(ext) for ext in model_exts)]
onnx_binaries = collect_dynamic_libs("onnxruntime")
cv2_binaries = collect_dynamic_libs("cv2")
rapidocr_datas = [d for d in collect_data_files("rapidocr_onnxruntime") if not any(str(d[0]).lower().endswith(ext) for ext in model_exts)]

a = Analysis(
    [str(repo / "auto_main.py")],
    pathex=[str(repo)],
    binaries=onnx_binaries + cv2_binaries,
    datas=[
        (str(repo / "assets"), "assets"),
        (str(repo / "toolrecap_v2" / "voice" / "omnivoice_adapter.py"), "toolrecap_v2/voice"),
        (str(repo / "toolrecap_v2" / "voice" / "bootstrap.py"), "toolrecap_v2/voice"),
    ]
    + piper_datas
    + onnx_datas
    + rapidocr_datas,
    hiddenimports=[
        "PIL",
        "PIL.Image",
        "soundfile",
        "onnxruntime",
        "faster_whisper",
        "rapidocr_onnxruntime",
        "cv2",
        "pyclipper",
        "shapely",
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
        "toolrecap_v2.renderer",
        "toolrecap_v2.output_validation",
        "toolrecap_v2.api_client",
        "toolrecap_v2.audio_mix",
        "toolrecap_v2.analyzer",
        "toolrecap_v2.analyzer.connection",
        "toolrecap_v2.analyzer.engine",
        "toolrecap_v2.analyzer.errors",
        "toolrecap_v2.analyzer.evidence",
        "toolrecap_v2.analyzer.finalizer",
        "toolrecap_v2.analyzer.phases",
        "toolrecap_v2.analyzer.prompts",
        "toolrecap_v2.domain",
        "toolrecap_v2.domain.cache",
        "toolrecap_v2.domain.enums",
        "toolrecap_v2.domain.models",
        "toolrecap_v2.domain.title",
        "toolrecap_v2.subtitles",
        "toolrecap_v2.subtitles.cache",
        "toolrecap_v2.subtitles.discovery",
        "toolrecap_v2.subtitles.models",
        "toolrecap_v2.subtitles.ocr",
        "toolrecap_v2.subtitles.parsers",
        "toolrecap_v2.subtitles.pgs",
        "toolrecap_v2.subtitles.pipeline",
        "toolrecap_v2.subtitles.remap",
        "toolrecap_v2.subtitles.vobsub",
        "toolrecap_v2.voice",
        "toolrecap_v2.voice.audio_mix",
        "toolrecap_v2.voice.audio_preview",
        "toolrecap_v2.voice.bootstrap",
        "toolrecap_v2.voice.catalog",
        "toolrecap_v2.voice.manager",
        "toolrecap_v2.voice.omnivoice_adapter",
        "toolrecap_v2.voice.voice_updater",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests", "pytest", "_pytest", "unittest", ".prime"],
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

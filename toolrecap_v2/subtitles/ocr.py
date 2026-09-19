"""RapidOCR ONNX adapter, local model manager, quality gate, and AI vision fallback."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..paths import default_data_directory
from .parsers import strip_formatting_tags


ProgressCallback = Callable[[int, int, str], None]
AiVisionFallback = Callable[[Image.Image], str | None]


# Pinned model metadata for RapidOCR PP-OCRv4 (official ModelScope distribution)
PINNED_MODELS: dict[str, dict[str, str]] = {
    "det": {
        "filename": "ch_PP-OCRv4_det_mobile.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
        "sha256": "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
    },
    "rec": {
        "filename": "ch_PP-OCRv4_rec_mobile.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
        "sha256": "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
    },
    "cls": {
        "filename": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "sha256": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    },
}


@dataclass
class OcrResult:
    text: str
    confidence: float
    is_valid: bool
    reason: str = ""
    source: str = "rapidocr"  # 'rapidocr', 'ai_gateway', 'empty', 'rejected'


class OcrModelManager:
    """Manages local RapidOCR ONNX model files under LOCALAPPDATA with hash validation and pinning."""

    def __init__(self, model_dir: Path | None = None) -> None:
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = default_data_directory() / "models" / "ocr"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_model_paths(self) -> dict[str, Path]:
        """Return dict of model file paths."""
        return {
            key: self.model_dir / meta["filename"]
            for key, meta in PINNED_MODELS.items()
        }

    def are_models_available(self) -> bool:
        """Check if all pinned model files exist on disk with non-zero size."""
        return all(p.is_file() and p.stat().st_size > 0 for p in self.get_model_paths().values())

    def verify_hashes(self) -> bool:
        """Validate SHA-256 hashes of local model files."""
        for key, meta in PINNED_MODELS.items():
            path = self.model_dir / meta["filename"]
            if not path.is_file():
                return False
            expected_hash = meta.get("sha256")
            if expected_hash:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                if h.hexdigest().lower() != expected_hash.lower():
                    return False
        return True

    def download_models(
        self,
        progress_callback: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Download pinned models atomically with visible byte progress and cancellation."""
        import urllib.request

        paths = self.get_model_paths()

        for key, meta in PINNED_MODELS.items():
            if cancel_check and cancel_check():
                raise RuntimeError("Tải mô hình OCR bị hủy.")

            dest = paths[key]
            expected_hash = meta.get("sha256")

            # Warm cache check: verify existing file
            if dest.is_file():
                if expected_hash:
                    h = hashlib.sha256()
                    with dest.open("rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    if h.hexdigest().lower() == expected_hash.lower():
                        continue  # Cache hit, skip download
                else:
                    continue
                # Hash mismatch on existing file: remove and re-download
                dest.unlink(missing_ok=True)

            url = meta["url"]
            tmp_dest = dest.with_suffix(".tmp")

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ToolRecap/2.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    total_bytes = int(resp.headers.get("Content-Length") or 0)
                    downloaded = 0
                    hasher = hashlib.sha256()

                    with tmp_dest.open("wb") as out_f:
                        while True:
                            if cancel_check and cancel_check():
                                raise RuntimeError("Tải mô hình OCR bị hủy.")
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(
                                    downloaded,
                                    total_bytes,
                                    f"Đang tải {meta['filename']} ({downloaded}/{total_bytes} bytes)",
                                )

                # Verify hash of downloaded file
                actual_hash = hasher.hexdigest().lower()
                if expected_hash and actual_hash != expected_hash.lower():
                    tmp_dest.unlink(missing_ok=True)
                    raise ValueError(
                        f"Sai mã băm cho {meta['filename']}: nhận {actual_hash}, mong đợi {expected_hash}"
                    )

                # Atomic rename
                os.replace(tmp_dest, dest)

                if progress_callback:
                    progress_callback(downloaded, total_bytes, f"Đã tải {meta['filename']}")

            except Exception:
                if tmp_dest.is_file():
                    tmp_dest.unlink(missing_ok=True)
                raise

        return self.verify_hashes()


class OcrAdapter:
    """RapidOCR ONNX adapter with lazy loading, quality gating, and strict AI fallback."""

    MIN_CONFIDENCE = 0.50

    def __init__(
        self,
        model_manager: OcrModelManager | None = None,
        custom_engine: Any = None,
    ) -> None:
        self.model_manager = model_manager or OcrModelManager()
        self._engine = custom_engine
        self._initialized = custom_engine is not None

    @classmethod
    def is_package_installed(cls) -> bool:
        """Check if rapidocr package is installed and importable."""
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            try:
                import rapidocr  # noqa: F401
                return True
            except ImportError:
                return False

    def ensure_models(
        self,
        progress_callback: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        """If package is installed and models missing, download them. Cache warm no download."""
        if not self.is_package_installed():
            return False
        if self.model_manager.are_models_available() or self._has_bundled_models():
            return True
        return self.model_manager.download_models(
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _has_bundled_models(self) -> bool:
        """Check if rapidocr package provides bundled models."""
        try:
            import rapidocr_onnxruntime
            pkg_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"
            if (pkg_dir / "ch_PP-OCRv4_rec_infer.onnx").is_file():
                return True
        except ImportError:
            pass
        try:
            import rapidocr
            pkg_dir = Path(rapidocr.__file__).parent / "models"
            if (pkg_dir / "PP-OCRv6_rec_small.onnx").is_file():
                return True
        except ImportError:
            pass
        return False

    def is_engine_ready(self) -> bool:
        """Check if OCR engine can be loaded."""
        if self._initialized:
            return True
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return self.model_manager.are_models_available() or self._has_bundled_models()
        except ImportError:
            try:
                import rapidocr  # noqa: F401
                return self.model_manager.are_models_available() or self._has_bundled_models()
            except ImportError:
                return False

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError(
                    "Gói 'rapidocr_onnxruntime' hoặc 'rapidocr' chưa được cài đặt. OCR bitmap không khả dụng."
                ) from exc

        if self.model_manager.are_models_available():
            model_paths = self.model_manager.get_model_paths()
            self._engine = RapidOCR(
                det_model_path=str(model_paths["det"]),
                rec_model_path=str(model_paths["rec"]),
                cls_model_path=str(model_paths["cls"]),
            )
        elif self._has_bundled_models():
            self._engine = RapidOCR()
        else:
            raise RuntimeError(
                "Không tìm thấy mô hình OCR và gói không có mô hình tích hợp. Hãy tải mô hình trước."
            )

        self._initialized = True
        return self._engine

    @classmethod
    def quality_gate(cls, text: str, confidence: float) -> tuple[bool, str]:
        """Check if OCR text passes quality gate. Returns (is_valid, reason)."""
        clean = strip_formatting_tags(text).strip()
        if not clean:
            return False, "Văn bản rỗng sau khi chuẩn hóa."

        if confidence < cls.MIN_CONFIDENCE:
            return False, f"Độ tin cậy thấp ({confidence:.2f} < {cls.MIN_CONFIDENCE:.2f})."

        # Check for presence of alphanumeric words
        has_alnum = bool(re.search(r"[a-zA-Z0-9\u00C0-\u024F\u1EA0-\u1EF9]", clean))
        if not has_alnum:
            return False, "Không có ký tự chữ hoặc số hợp lệ."

        # Check for repetitive garbage sequences (e.g. '||||||||' or '-------')
        if len(clean) >= 6 and len(set(clean.replace(" ", ""))) <= 2:
            return False, "Chuỗi ký tự lặp rác."

        return True, "Hợp lệ"

    def ocr_image(
        self,
        image: Image.Image,
        *,
        ai_fallback_fn: AiVisionFallback | None = None,
        vision_supported: bool = False,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OcrResult:
        """Perform OCR on a cropped subtitle bounding box. Never pass full video frames.

        If local OCR fails or yields low-confidence text:
        - Calls ai_fallback_fn ONLY if caller explicitly declared vision_supported=True.
        - Never pretends or invents text.
        """
        import numpy as np

        if cancel_check and cancel_check():
            raise RuntimeError("OCR phụ đề đã bị hủy.")

        # Strict sanity invariant: Never OCR full 1080p / 4K frames
        if image.width >= 1900 and image.height >= 1000:
            raise ValueError(
                f"Vi phạm nguyên tắc bounding box: Kích thước ảnh {image.width}x{image.height} quá lớn (toàn khung hình)."
            )

        # 1. Attempt local RapidOCR
        raw_text = ""
        avg_score = 0.0
        ocr_failed = False

        try:
            engine = self._get_engine()
            # Convert PIL.Image to numpy array (expected by OpenCV/RapidOCR)
            img_arr = np.array(image.convert("RGB"))
            raw_output = engine(img_arr)

            texts: list[str] = []
            scores: list[float] = []

            # Modern RapidOCROutput object with txts and scores attributes
            if hasattr(raw_output, "txts") and hasattr(raw_output, "scores"):
                if raw_output.txts and raw_output.scores:
                    texts = [str(t) for t in raw_output.txts]
                    scores = [float(s) for s in raw_output.scores]
            else:
                # Legacy tuple API: (result, elapse) or list
                items = raw_output[0] if isinstance(raw_output, tuple) else raw_output
                if items and isinstance(items, (list, tuple)):
                    for item in items:
                        # Item structure: [box, text, score]
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            texts.append(str(item[1]))
                            scores.append(float(item[2]))

            if texts:
                raw_text = " ".join(texts)
                avg_score = sum(scores) / len(scores) if scores else 0.0
        except Exception:
            ocr_failed = True

        is_valid, reason = self.quality_gate(raw_text, avg_score)
        if is_valid and not ocr_failed:
            return OcrResult(
                text=strip_formatting_tags(raw_text),
                confidence=avg_score,
                is_valid=True,
                reason="Local RapidOCR thành công",
                source="rapidocr",
            )

        # 2. Local OCR failed or was rejected by quality gate: AI Gateway Vision Fallback
        if cancel_check and cancel_check():
            raise RuntimeError("OCR phụ đề đã bị hủy.")
        if ai_fallback_fn is not None and vision_supported:
            try:
                ai_text = ai_fallback_fn(image)
                if ai_text:
                    ai_clean = strip_formatting_tags(ai_text).strip()
                    if ai_clean:
                        return OcrResult(
                            text=ai_clean,
                            confidence=0.85,
                            is_valid=True,
                            reason="AI Vision Gateway fallback thành công",
                            source="ai_gateway",
                        )
            except Exception:
                pass

        # 3. Caller did not declare vision_supported or AI fallback returned empty:
        # Strict invariant: No pretending. Never invent text.
        return OcrResult(
            text="",
            confidence=0.0,
            is_valid=False,
            reason=f"OCR không đạt chất lượng ({reason}) và Vision AI không được khai báo/hỗ trợ.",
            source="rejected",
        )

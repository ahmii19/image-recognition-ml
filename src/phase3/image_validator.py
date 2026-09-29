"""Robust image validation, format normalization, and metadata inspection for Phase 3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.phase3.config import (
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    SUPPORTED_IMAGE_EXTENSIONS,
)


@dataclass
class ImageMetadata:
    """Metadata extracted from validated image."""
    file_path: str
    file_name: str
    format: str
    width: int
    height: int
    channels: int
    aspect_ratio: float
    file_size_bytes: int


@dataclass
class ValidationResult:
    """Normalized output from image validator."""
    success: bool
    image: Optional[Image.Image] = None
    np_array: Optional[np.ndarray] = None
    metadata: Optional[ImageMetadata] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to JSON-serializable dictionary."""
        if not self.success:
            return {
                "success": False,
                "error": self.error,
                "error_code": self.error_code,
            }
        return {
            "success": True,
            "metadata": {
                "file_path": self.metadata.file_path if self.metadata else "",
                "file_name": self.metadata.file_name if self.metadata else "",
                "format": self.metadata.format if self.metadata else "",
                "width": self.metadata.width if self.metadata else 0,
                "height": self.metadata.height if self.metadata else 0,
                "channels": self.metadata.channels if self.metadata else 0,
                "aspect_ratio": round(self.metadata.aspect_ratio, 3) if self.metadata else 0.0,
                "file_size_bytes": self.metadata.file_size_bytes if self.metadata else 0,
            },
        }


def validate_and_load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
) -> ValidationResult:
    """Load, validate, normalize, and inspect image input.

    Performs rigorous sanity checks:
    - Path existence and read permissions
    - Zero-byte file detection
    - Extension verification (.jpg, .jpeg, .png, .webp, .bmp)
    - Image corruption / decompression bomb checks
    - Dimensional boundary checks (e.g. 10px to 8192px)
    - Safe RGB conversion (handling Grayscale 'L', Palette 'P', RGBA 'RGBA', CMYK 'CMYK')
    - Alpha compositing on pure white background to avoid dark transparency artifacts
    - EXIF orientation auto-rotation

    Args:
        image_input: File path (str/Path), PIL Image, or Numpy Array.

    Returns:
        ValidationResult with validated PIL RGB Image, uint8 Numpy array, and metadata.
    """
    # 1. Handle PIL Image directly
    if isinstance(image_input, Image.Image):
        try:
            return _normalize_pil_image(image_input, source_path="<in-memory-pil>")
        except Exception as e:
            return ValidationResult(
                success=False,
                error=f"Failed to process in-memory PIL image: {str(e)}",
                error_code="PIL_PROCESSING_ERROR",
            )

    # 2. Handle Numpy Array directly
    if isinstance(image_input, np.ndarray):
        try:
            return _normalize_numpy_image(image_input)
        except Exception as e:
            return ValidationResult(
                success=False,
                error=f"Failed to process in-memory NumPy array: {str(e)}",
                error_code="NUMPY_PROCESSING_ERROR",
            )

    # 3. Handle File Path
    try:
        path = Path(image_input)
    except Exception as e:
        return ValidationResult(
            success=False,
            error=f"Invalid file path specification: {image_input}. Error: {str(e)}",
            error_code="INVALID_PATH",
        )

    if not path.exists():
        return ValidationResult(
            success=False,
            error=f"Image file does not exist at path: {path}",
            error_code="FILE_NOT_FOUND",
        )

    if not path.is_file():
        return ValidationResult(
            success=False,
            error=f"Specified path is a directory, not a file: {path}",
            error_code="PATH_IS_DIRECTORY",
        )

    try:
        file_size = path.stat().st_size
    except Exception as e:
        return ValidationResult(
            success=False,
            error=f"Could not read file attributes: {str(e)}",
            error_code="FILE_STAT_ERROR",
        )

    if file_size == 0:
        return ValidationResult(
            success=False,
            error=f"Image file is completely empty (0 bytes): {path}",
            error_code="EMPTY_FILE",
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        return ValidationResult(
            success=False,
            error=(
                f"Unsupported image format '{suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            ),
            error_code="UNSUPPORTED_FORMAT",
        )

    try:
        with Image.open(path) as raw_img:
            # Check for corrupted files
            raw_img.verify()

        # Reopen after verify()
        with Image.open(path) as raw_img:
            raw_img.load()
            return _normalize_pil_image(raw_img, source_path=str(path), file_size_bytes=file_size)

    except UnidentifiedImageError:
        return ValidationResult(
            success=False,
            error=f"File is corrupted or not a recognized image format: {path}",
            error_code="CORRUPTED_IMAGE",
        )
    except Exception as e:
        return ValidationResult(
            success=False,
            error=f"Failed to open or decode image: {str(e)}",
            error_code="IMAGE_DECODE_ERROR",
        )


def _normalize_pil_image(
    pil_img: Image.Image,
    source_path: str = "",
    file_size_bytes: int = 0,
) -> ValidationResult:
    """Normalize PIL Image into standardized 3-channel RGB with correct orientation."""
    # Auto-rotate according to EXIF orientation tag if present
    try:
        oriented_img = ImageOps.exif_transpose(pil_img)
        if oriented_img is not None:
            pil_img = oriented_img
    except Exception:
        pass

    width, height = pil_img.size

    # Validate dimensions
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return ValidationResult(
            success=False,
            error=f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({MIN_IMAGE_DIMENSION}px).",
            error_code="IMAGE_TOO_SMALL",
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return ValidationResult(
            success=False,
            error=f"Image dimensions ({width}x{height}) exceed maximum allowable size ({MAX_IMAGE_DIMENSION}px).",
            error_code="IMAGE_TOO_LARGE",
        )

    # Safe RGB conversion handling Alpha transparency
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        # Composite onto pure white canvas to avoid black transparency artifacts
        rgba_img = pil_img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        rgb_img = Image.alpha_composite(white_bg, rgba_img).convert("RGB")
    elif pil_img.mode != "RGB":
        rgb_img = pil_img.convert("RGB")
    else:
        rgb_img = pil_img.copy()

    np_array = np.array(rgb_img, dtype=np.uint8)
    aspect_ratio = float(width / max(1, height))
    file_name = Path(source_path).name if source_path and source_path != "<in-memory-pil>" else "in_memory_image"
    img_format = pil_img.format if pil_img.format else "RGB"

    metadata = ImageMetadata(
        file_path=source_path,
        file_name=file_name,
        format=img_format,
        width=width,
        height=height,
        channels=3,
        aspect_ratio=aspect_ratio,
        file_size_bytes=file_size_bytes,
    )

    return ValidationResult(
        success=True,
        image=rgb_img,
        np_array=np_array,
        metadata=metadata,
    )


def _normalize_numpy_image(arr: np.ndarray) -> ValidationResult:
    """Normalize a raw NumPy array into a validated RGB PIL image."""
    if arr.ndim == 2:  # Grayscale (H, W)
        h, w = arr.shape
        c = 1
        arr_rgb = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c == 1:
            arr_rgb = np.concatenate([arr] * 3, axis=-1)
        elif c == 3:
            arr_rgb = arr
        elif c == 4:
            # Drop alpha or composite
            arr_rgb = arr[:, :, :3]
        else:
            return ValidationResult(
                success=False,
                error=f"Unsupported number of channels: {c}. Expected 1, 3, or 4.",
                error_code="INVALID_CHANNELS",
            )
    else:
        return ValidationResult(
            success=False,
            error=f"NumPy array has invalid rank {arr.ndim}. Expected 2D or 3D image tensor.",
            error_code="INVALID_ARRAY_RANK",
        )

    # Ensure uint8 [0, 255]
    if arr_rgb.dtype != np.uint8:
        if np.max(arr_rgb) <= 1.0 and np.min(arr_rgb) >= 0.0:
            arr_rgb = (arr_rgb * 255.0).astype(np.uint8)
        else:
            arr_rgb = np.clip(arr_rgb, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_rgb, mode="RGB")
    return _normalize_pil_image(pil_img, source_path="<in-memory-numpy>")

"""Centralized Runtime Device Resolution and Hardware Diagnostic Utilities.

Provides unified device targeting across PyTorch vision models (OpenCLIP, OWL-ViT,
RegionRecognitionPipeline) with support for 'auto', 'cpu', and 'cuda' execution.
"""

import logging
import os
from typing import Any, Dict, Optional, Union
import torch

logger = logging.getLogger("src.device")


def resolve_device(
    device_target: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Resolve target execution device from explicit parameter or ML_DEVICE environment variable.

    Supported Modes:
        - 'auto': Automatically select CUDA if available on host; fallback gracefully to CPU.
        - 'cpu': Strictly force CPU execution.
        - 'cuda' or 'cuda:X': Require CUDA GPU. Raises RuntimeError if CUDA is unavailable.

    Args:
        device_target: Optional device string ('auto', 'cpu', 'cuda', 'cuda:0') or torch.device.
                       If None, resolves from os.environ['ML_DEVICE'] (default: 'auto').

    Returns:
        Resolved torch.device instance.

    Raises:
        RuntimeError: If CUDA is explicitly requested ('cuda' / 'cuda:X') but CUDA is not available.
        ValueError: If an unsupported device specification string is provided.
    """
    if isinstance(device_target, torch.device):
        return device_target

    # Determine raw device specification string
    if device_target is None:
        raw_spec = os.getenv("ML_DEVICE", "auto").strip()
    else:
        raw_spec = str(device_target).strip()

    normalized_spec = raw_spec.lower()

    if normalized_spec in ("auto", "default"):
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected. Automatically selected device: cuda")
            return torch.device("cuda")
        logger.info("CUDA unavailable on host. Automatically selected device: cpu")
        return torch.device("cpu")

    if normalized_spec == "cpu":
        return torch.device("cpu")

    if normalized_spec == "cuda" or normalized_spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            err_msg = (
                f"CUDA execution was explicitly requested ('{raw_spec}'), but CUDA is not available "
                f"on this host machine. Current PyTorch build does not detect an accessible NVIDIA GPU. "
                f"To run on CPU, configure ML_DEVICE=auto or ML_DEVICE=cpu."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return torch.device(normalized_spec)

    # If an unrecognized string format was provided
    raise ValueError(
        f"Unsupported device target '{raw_spec}'. Supported options: 'auto', 'cpu', 'cuda', or 'cuda:<index>'."
    )


def get_device_info() -> Dict[str, Any]:
    """Retrieve runtime device diagnostic metadata without throwing exceptions.

    Returns:
        Structured dictionary containing selected device, CUDA availability, GPU name,
        PyTorch version, and configured environment setting.
    """
    cuda_avail = bool(torch.cuda.is_available())
    configured = os.getenv("ML_DEVICE", "auto")

    try:
        current_dev = str(resolve_device())
    except Exception as e:
        current_dev = f"error: {str(e)}"

    device_name: Optional[str] = None
    device_count: int = 0

    if cuda_avail:
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "Unknown CUDA Device"

    return {
        "selected_device": current_dev,
        "cuda_available": cuda_avail,
        "cuda_device_count": device_count,
        "cuda_device_name": device_name,
        "pytorch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "configured_setting": configured,
    }

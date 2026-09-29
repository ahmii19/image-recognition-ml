"""Configuration constants and environment overrides for Phase 7 VLM engine.

All settings are controllable via environment variables, enabling zero-code
configuration changes between local CPU development and Colab CUDA deployment.

Environment Variables:
    VLM_MODEL_ID          — HuggingFace model identifier (default: Qwen/Qwen2.5-VL-3B-Instruct)
    VLM_DEVICE            — Device target: 'auto' | 'cpu' | 'cuda' | 'cuda:0' (default: auto)
    VLM_TORCH_THREADS     — CPU intra-op thread count when running on CPU (default: 4)
    VLM_MAX_NEW_TOKENS    — Maximum tokens the VLM may generate per response (default: 512)
    VLM_MIN_NEW_TOKENS    — Minimum tokens the VLM must generate per response (default: 1)
    VLM_TEMPERATURE       — Sampling temperature (default: 0.1 for deterministic output)
    VLM_DO_SAMPLE         — Enable sampling-based generation (default: false = greedy)
    VLM_AUTO_UNLOAD       — Enable automatic idle-based unload (default: false)
    VLM_IDLE_UNLOAD_SECS  — Seconds of inactivity before auto-unload fires (default: 600)
    VLM_LOAD_IN_4BIT      — Enable bitsandbytes 4-bit quantization (default: false)
    VLM_LOAD_IN_8BIT      — Enable bitsandbytes 8-bit quantization (default: false)
    VLM_TRUST_REMOTE_CODE — Trust remote code in model hub (required for Qwen) (default: true)
    VLM_MAX_IMAGE_SIZE    — Maximum single-side pixel length before resize (default: 1280)
"""

import os
from typing import Final

# ── Model Identity ────────────────────────────────────────────────────────────
VLM_MODEL_ID: Final[str] = os.getenv(
    "VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"
)

# ── Device Resolution ─────────────────────────────────────────────────────────
# Falls back to global ML_DEVICE env, then to 'auto'
VLM_DEVICE: Final[str] = os.getenv(
    "VLM_DEVICE", os.getenv("ML_DEVICE", "auto")
)

# ── CPU Thread Budget ─────────────────────────────────────────────────────────
VLM_TORCH_THREADS: Final[int] = int(os.getenv("VLM_TORCH_THREADS", "4"))

# ── Generation Parameters ─────────────────────────────────────────────────────
VLM_MAX_NEW_TOKENS: Final[int] = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))
VLM_MIN_NEW_TOKENS: Final[int] = int(os.getenv("VLM_MIN_NEW_TOKENS", "1"))
VLM_TEMPERATURE: Final[float] = float(os.getenv("VLM_TEMPERATURE", "0.1"))
VLM_DO_SAMPLE: Final[bool] = os.getenv("VLM_DO_SAMPLE", "false").lower() in (
    "true",
    "1",
    "yes",
)

# ── Memory / Quantization ─────────────────────────────────────────────────────
VLM_LOAD_IN_4BIT: Final[bool] = os.getenv("VLM_LOAD_IN_4BIT", "false").lower() in (
    "true",
    "1",
    "yes",
)
VLM_LOAD_IN_8BIT: Final[bool] = os.getenv("VLM_LOAD_IN_8BIT", "false").lower() in (
    "true",
    "1",
    "yes",
)

# ── Safety / Hub ──────────────────────────────────────────────────────────────
VLM_TRUST_REMOTE_CODE: Final[bool] = os.getenv(
    "VLM_TRUST_REMOTE_CODE", "true"
).lower() in ("true", "1", "yes")

# ── Image Pre-processing ──────────────────────────────────────────────────────
# Images larger than this (on any side) are resized before being passed to the VLM
VLM_MAX_IMAGE_SIZE: Final[int] = int(os.getenv("VLM_MAX_IMAGE_SIZE", "1280"))

# ── Lifecycle ─────────────────────────────────────────────────────────────────
VLM_AUTO_UNLOAD: Final[bool] = os.getenv("VLM_AUTO_UNLOAD", "false").lower() in (
    "true",
    "1",
    "yes",
)
VLM_IDLE_UNLOAD_SECONDS: Final[int] = int(os.getenv("VLM_IDLE_UNLOAD_SECS", "600"))

# ── Structured-Output Prompt System ──────────────────────────────────────────
# System instruction injected at the start of every chat template
VLM_SYSTEM_PROMPT: Final[str] = (
    "You are a world-class visual AI assistant. "
    "When asked to describe an image, always respond with a single valid JSON object. "
    "Do not include markdown code fences, explanatory text, or any content outside the JSON object. "
    "Use double-quoted keys and string values. "
    "The JSON must contain exactly the fields requested by the user prompt."
)

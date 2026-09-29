"""Phase 7 — General Image Understanding via Vision-Language Model (VLM).

This package implements a production-grade General Image Understanding engine
using Qwen2.5-VL-3B-Instruct as the primary VLM candidate.

Sub-modules:
    config   — VLM configuration constants and environment overrides.
    engine   — Core VLMEngine: prompt building, inference, structured response parsing.
    lifecycle — Thread-safe lazy singleton lifecycle manager with auto-unload support.

Architecture:
    - Fully isolated from Phase 1–6 modules (no cross-phase imports from Phase 7).
    - Reuses src.device for unified device resolution.
    - CPU and CUDA execution both supported; CUDA path targets Colab T4 (16 GB VRAM).
    - Lazy initialization: model is NOT loaded until first inference call.
    - Explicit unload: VLMLifecycleManager.unload() releases GPU/CPU memory on demand.
"""

from src.phase7.engine import VLMEngine
from src.phase7.lifecycle import VLMLifecycleManager

__all__ = [
    "VLMEngine",
    "VLMLifecycleManager",
]

"""Phase 7 VLM Lifecycle Manager — Thread-safe singleton with auto-unload support.

Mirrors the OWLViTLifecycleManager pattern from Phase 6 (src/phase6/owlvit/lifecycle.py)
and extends it with VLM-specific telemetry and configurable idle-based auto-unload.

Features:
    - Double-checked locking for thread-safe lazy initialization.
    - Automatic idle-based unloading via a daemon Timer thread.
    - Full telemetry: load count, total inference count, last access timestamp.
    - Explicit unload() for on-demand GPU/CPU memory reclamation.
    - get_status() for monitoring and health-check endpoints.

Usage:
    from src.phase7.lifecycle import VLMLifecycleManager

    engine = VLMLifecycleManager.get_engine()
    result = engine.understand(image_path)

    # Explicit unload (e.g., after a Colab inference batch)
    VLMLifecycleManager.unload()
"""

import gc
import logging
import threading
import time
from typing import Optional

import torch

from src.device import resolve_device
from src.phase7.config import (
    VLM_AUTO_UNLOAD,
    VLM_DEVICE,
    VLM_IDLE_UNLOAD_SECONDS,
    VLM_LOAD_IN_4BIT,
    VLM_LOAD_IN_8BIT,
    VLM_MODEL_ID,
    VLM_TORCH_THREADS,
    VLM_TRUST_REMOTE_CODE,
)
from src.phase7.engine import VLMEngine

logger = logging.getLogger("phase7.lifecycle")


class VLMLifecycleManager:
    """Thread-safe singleton lifecycle manager for the VLMEngine.

    Class-level state ensures only one VLMEngine instance exists at any time,
    regardless of how many threads call get_engine() concurrently.

    Auto-unload:
        When VLM_AUTO_UNLOAD is True, an idle timer is reset on every inference.
        If no inference occurs within VLM_IDLE_UNLOAD_SECONDS, the model is
        automatically unloaded from memory to reclaim GPU/CPU resources.
    """

    _instance_lock: threading.Lock = threading.Lock()
    _engine: Optional[VLMEngine] = None
    _last_accessed_at: Optional[float] = None
    _total_inferences: int = 0
    _load_count: int = 0
    _idle_timer: Optional[threading.Timer] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def is_loaded(cls) -> bool:
        """Return True if the VLMEngine singleton is currently resident in memory."""
        return cls._engine is not None and cls._engine.is_loaded()

    @classmethod
    def get_engine(cls) -> VLMEngine:
        """Return the singleton VLMEngine, loading it lazily on first call.

        Uses double-checked locking to ensure thread-safety without acquiring
        the lock on every call after the engine is initialized.

        Returns:
            Loaded VLMEngine singleton.
        """
        if cls._engine is None:
            with cls._instance_lock:
                if cls._engine is None:
                    target_device = str(resolve_device(VLM_DEVICE))
                    logger.info(
                        "Lazy-initializing VLMEngine singleton on device '%s' "
                        "(model: %s)...",
                        target_device,
                        VLM_MODEL_ID,
                    )
                    cls._engine = VLMEngine(
                        model_id=VLM_MODEL_ID,
                        device=target_device,
                        torch_threads=VLM_TORCH_THREADS,
                        load_in_4bit=VLM_LOAD_IN_4BIT,
                        load_in_8bit=VLM_LOAD_IN_8BIT,
                        trust_remote_code=VLM_TRUST_REMOTE_CODE,
                    )
                    cls._load_count += 1
                    logger.info(
                        "VLMEngine singleton initialized (load #%d).", cls._load_count
                    )

        cls._last_accessed_at = time.time()
        cls._total_inferences += 1

        if VLM_AUTO_UNLOAD and VLM_IDLE_UNLOAD_SECONDS > 0:
            cls._schedule_idle_unload()

        return cls._engine

    @classmethod
    def unload(cls) -> bool:
        """Safely release the VLMEngine singleton and reclaim GPU/CPU memory.

        Cancels any pending idle-unload timer before releasing the model.

        Returns:
            True if an active engine was unloaded, False if already unloaded.
        """
        with cls._instance_lock:
            # Cancel any pending idle timer
            if cls._idle_timer is not None:
                cls._idle_timer.cancel()
                cls._idle_timer = None

            if cls._engine is None:
                logger.debug("VLM unload requested but engine is already unloaded.")
                return False

            logger.info(
                "Unloading VLMEngine singleton (was loaded %d time(s), "
                "%d total inferences)...",
                cls._load_count,
                cls._total_inferences,
            )

            # Delegate internal memory release to the engine
            cls._engine.unload()
            del cls._engine
            cls._engine = None

            # Collect Python cyclic references
            gc.collect()

            # Release PyTorch CUDA cache if GPU is available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA memory cache cleared after VLM lifecycle unload.")
                except Exception as exc:
                    logger.warning("torch.cuda.empty_cache() failed: %s", exc)

            logger.info("VLMEngine singleton unloaded successfully.")
            return True

    @classmethod
    def get_status(cls) -> dict:
        """Return runtime lifecycle telemetry for health checks and diagnostics.

        Returns:
            Dictionary with: loaded, model_id, device, load_count, total_inferences,
            last_accessed_at, auto_unload_enabled, idle_unload_seconds,
            load_in_4bit, load_in_8bit, param_count, load_time_s.
        """
        engine_info: dict = {}
        if cls._engine is not None:
            try:
                engine_info = cls._engine.get_info()
            except Exception:
                pass

        resolved_device = str(resolve_device(VLM_DEVICE))

        return {
            "loaded": cls.is_loaded(),
            "model_id": VLM_MODEL_ID,
            "device": engine_info.get("device", resolved_device),
            "load_count": cls._load_count,
            "total_inferences": cls._total_inferences,
            "last_accessed_at": cls._last_accessed_at,
            "auto_unload_enabled": VLM_AUTO_UNLOAD,
            "idle_unload_seconds": VLM_IDLE_UNLOAD_SECONDS,
            "load_in_4bit": VLM_LOAD_IN_4BIT,
            "load_in_8bit": VLM_LOAD_IN_8BIT,
            "param_count": engine_info.get("param_count", 0),
            "load_time_s": engine_info.get("load_time_s", 0.0),
            "lazy_loaded": True,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @classmethod
    def _schedule_idle_unload(cls) -> None:
        """(Re)schedule automatic model unload after an idle period.

        Called internally after every inference when auto-unload is enabled.
        Each new inference resets the idle countdown.
        """
        if cls._idle_timer is not None:
            cls._idle_timer.cancel()

        def _do_unload() -> None:
            logger.info(
                "VLM idle timer expired (%ds). Triggering automatic unload...",
                VLM_IDLE_UNLOAD_SECONDS,
            )
            cls.unload()

        cls._idle_timer = threading.Timer(VLM_IDLE_UNLOAD_SECONDS, _do_unload)
        cls._idle_timer.daemon = True
        cls._idle_timer.start()
        logger.debug(
            "VLM idle-unload timer scheduled: %ds.", VLM_IDLE_UNLOAD_SECONDS
        )

import gc
import logging
import threading
import time
from typing import Optional
import torch

from src.device import resolve_device
from src.phase6.owlvit.config import (
    AUTO_UNLOAD_ENABLED,
    IDLE_UNLOAD_SECONDS,
    OWL_VIT_DEVICE,
    OWL_VIT_MODEL_ID,
    OWL_VIT_TORCH_THREADS,
)
from src.phase6.owlvit.detector import OWLViTDetector

logger = logging.getLogger("phase6.owlvit.lifecycle")


class OWLViTLifecycleManager:
    """Manages lazy instantiation, concurrency synchronization, and memory release for OWL-ViT."""

    _instance_lock = threading.Lock()
    _detector: Optional[OWLViTDetector] = None
    _last_accessed_at: Optional[float] = None
    _total_inferences: int = 0
    _load_count: int = 0
    _idle_timer: Optional[threading.Timer] = None

    @classmethod
    def is_loaded(cls) -> bool:
        """Check whether the OWL-ViT model is currently resident in memory."""
        return cls._detector is not None

    @classmethod
    def get_detector(cls) -> OWLViTDetector:
        """Retrieve the singleton OWLViTDetector instance with thread-safe lazy loading.

        Returns:
            Instantiated OWLViTDetector singleton.
        """
        if cls._detector is None:
            with cls._instance_lock:
                if cls._detector is None:
                    target_device = str(resolve_device(OWL_VIT_DEVICE))
                    logger.info(
                        f"Triggering lazy initialization of OWLViTDetector singleton on {target_device}..."
                    )
                    cls._detector = OWLViTDetector(
                        model_id=OWL_VIT_MODEL_ID,
                        device=target_device,
                        torch_threads=OWL_VIT_TORCH_THREADS,
                    )
                    cls._load_count += 1

        cls._last_accessed_at = time.time()
        cls._total_inferences += 1

        if AUTO_UNLOAD_ENABLED and IDLE_UNLOAD_SECONDS > 0:
            cls._schedule_idle_unload()

        return cls._detector

    @classmethod
    def unload(cls) -> bool:
        """Safely release the OWL-ViT model from memory and trigger garbage collection.

        Returns:
            True if an active model was unloaded, False if already unloaded.
        """
        with cls._instance_lock:
            if cls._idle_timer:
                cls._idle_timer.cancel()
                cls._idle_timer = None

            if cls._detector is None:
                logger.debug("OWL-ViT unload requested, but model is already unloaded.")
                return False

            logger.info("Unloading OWLViTDetector singleton and reclaiming host/GPU memory...")
            del cls._detector.model
            del cls._detector.processor
            del cls._detector
            cls._detector = None

            # Collect cyclic Python references
            gc.collect()

            # Release PyTorch GPU memory cache if CUDA is active
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            logger.info("OWLViTDetector singleton unloaded successfully.")
            return True

    @classmethod
    def _schedule_idle_unload(cls) -> None:
        """Schedule automated model unloading after a period of inactivity."""
        if cls._idle_timer:
            cls._idle_timer.cancel()

        def _do_unload():
            logger.info(
                f"OWL-ViT idle timer expired ({IDLE_UNLOAD_SECONDS}s). Triggering auto-unload..."
            )
            cls.unload()

        cls._idle_timer = threading.Timer(IDLE_UNLOAD_SECONDS, _do_unload)
        cls._idle_timer.daemon = True
        cls._idle_timer.start()

    @classmethod
    def get_status(cls) -> dict:
        """Return runtime lifecycle telemetry for diagnostics and model metadata."""
        current_dev = cls._detector.device if cls._detector is not None else str(resolve_device(OWL_VIT_DEVICE))
        return {
            "loaded": cls.is_loaded(),
            "model_id": OWL_VIT_MODEL_ID,
            "device": current_dev,
            "torch_threads": OWL_VIT_TORCH_THREADS,
            "lazy_loaded": True,
            "load_count": cls._load_count,
            "total_inferences": cls._total_inferences,
            "last_accessed_at": cls._last_accessed_at,
            "auto_unload_enabled": AUTO_UNLOAD_ENABLED,
            "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        }

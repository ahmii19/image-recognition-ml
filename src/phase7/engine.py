"""Phase 7 VLM Engine — General Image Understanding.

Implements the core VLMEngine class, which:
    - Loads Qwen2.5-VL-3B-Instruct (or any VLM_MODEL_ID configured model).
    - Performs CPU and CUDA inference with device abstraction via src.device.
    - Supports 4-bit / 8-bit quantization (bitsandbytes) when configured.
    - Implements lazy loading: model is NOT loaded until first call to understand().
    - Implements explicit unload via unload() for GPU memory management.
    - Accepts PIL Images, file paths, URL strings, and numpy arrays as input.
    - Applies Phase 3 image validation for consistency with the existing pipeline.
    - Uses the structured prompt system (phase7.prompts) and structured response
      parser (phase7.parser) to return deterministic, JSON-aligned output.

NOTE: Do NOT call this class directly in production code — use VLMLifecycleManager
instead to get a thread-safe singleton with lifecycle management.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from src.device import resolve_device
from src.phase7.config import (
    VLM_DEVICE,
    VLM_DO_SAMPLE,
    VLM_LOAD_IN_4BIT,
    VLM_LOAD_IN_8BIT,
    VLM_MAX_IMAGE_SIZE,
    VLM_MAX_NEW_TOKENS,
    VLM_MIN_NEW_TOKENS,
    VLM_MODEL_ID,
    VLM_SYSTEM_PROMPT,
    VLM_TEMPERATURE,
    VLM_TORCH_THREADS,
    VLM_TRUST_REMOTE_CODE,
)
from src.phase7.parser import ParseResult, VLMParseError, parse_vlm_response
from src.phase7.prompts import PromptMode, VLMPrompt, build_custom_prompt, get_prompt

logger = logging.getLogger("phase7.engine")


def _resize_image_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    """Resize image so that neither width nor height exceeds max_side, preserving aspect ratio.

    Args:
        image: PIL Image to resize.
        max_side: Maximum allowed dimension in pixels.

    Returns:
        Resized PIL Image if needed, otherwise the original image unchanged.
    """
    w, h = image.size
    if w <= max_side and h <= max_side:
        return image
    scale = min(max_side / w, max_side / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.debug("Resizing image from (%d, %d) to (%d, %d) for VLM input.", w, h, new_w, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _load_image_input(
    image_input: Union[str, Path, Image.Image, "np.ndarray"],  # type: ignore[name-defined]
) -> Image.Image:
    """Normalise any supported image input type to a PIL RGB Image.

    Supports:
        - PIL.Image.Image instances
        - File path strings or Path objects
        - HTTP/HTTPS URL strings (requires requests + PIL URL support)
        - NumPy ndarray (HWC, uint8 or float32)

    Args:
        image_input: Image to load.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the input type is not supported.
        FileNotFoundError: If a file path does not exist.
        IOError: If the image file cannot be opened.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if path_str.startswith("http://") or path_str.startswith("https://"):
            import urllib.request
            from io import BytesIO
            with urllib.request.urlopen(path_str, timeout=15) as resp:
                data = resp.read()
            return Image.open(BytesIO(data)).convert("RGB")
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        return Image.open(p).convert("RGB")

    # NumPy array support
    try:
        import numpy as np
        if isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                arr = (image_input * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = image_input
            return Image.fromarray(arr).convert("RGB")
    except ImportError:
        pass

    raise ValueError(
        f"Unsupported image_input type: {type(image_input).__name__}. "
        "Supported: PIL.Image, file path string/Path, HTTP URL string, numpy ndarray."
    )


class VLMEngine:
    """General Image Understanding engine powered by Qwen2.5-VL-3B-Instruct.

    Provides structured image understanding across all visual categories:
    photographs, illustrations, documents, diagrams, charts, maps, medical images,
    and ambiguous/mixed-content images.

    Usage:
        engine = VLMEngine()
        result = engine.understand(image_path, mode=PromptMode.GENERAL)

    Or via the lifecycle manager (preferred for production):
        from src.phase7.lifecycle import VLMLifecycleManager
        engine = VLMLifecycleManager.get_engine()
        result = engine.understand(image_path)
    """

    def __init__(
        self,
        model_id: str = VLM_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        torch_threads: int = VLM_TORCH_THREADS,
        load_in_4bit: bool = VLM_LOAD_IN_4BIT,
        load_in_8bit: bool = VLM_LOAD_IN_8BIT,
        trust_remote_code: bool = VLM_TRUST_REMOTE_CODE,
    ) -> None:
        """Initialise the VLM engine and load the model + processor.

        Args:
            model_id: HuggingFace model identifier.
            device: Target execution device ('auto', 'cpu', 'cuda', etc.).
            torch_threads: CPU intra-op thread count (used when device is CPU).
            load_in_4bit: Enable 4-bit bitsandbytes quantization (requires CUDA + bitsandbytes).
            load_in_8bit: Enable 8-bit bitsandbytes quantization (requires CUDA + bitsandbytes).
            trust_remote_code: Trust remote model code (required for Qwen2.5-VL).
        """
        self.model_id = model_id
        self.device = str(resolve_device(device if device is not None else VLM_DEVICE))
        self.torch_threads = torch_threads
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.processor = None
        self._param_count: int = 0
        self._load_time_s: float = 0.0

        logger.info(
            "VLMEngine initialising: model_id=%s | device=%s | 4bit=%s | 8bit=%s",
            self.model_id, self.device, self.load_in_4bit, self.load_in_8bit,
        )

        if self.device == "cpu" and self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)
            logger.debug("CPU thread count set to %d.", self.torch_threads)

        self._load_model()

    def _load_model(self) -> None:
        """Load the VLM model and processor from HuggingFace hub or local cache."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        load_start = time.perf_counter()

        # Build quantization config if applicable
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if self.load_in_4bit else None,
                )
                logger.info(
                    "BitsAndBytes quantization config: 4bit=%s, 8bit=%s",
                    self.load_in_4bit, self.load_in_8bit,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not available; loading model in full precision. "
                    "Install bitsandbytes for quantization support."
                )
                quantization_config = None

        logger.info("Loading VLM processor from '%s'...", self.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        logger.info("Loading VLM model from '%s' on device '%s'...", self.model_id, self.device)

        # torch_dtype selection
        if self.device.startswith("cuda"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
        )

        # Move to device explicitly if device_map was not used (CPU path)
        if self.device == "cpu":
            self.model = self.model.to(torch.device("cpu"))

        self.model.eval()

        self._param_count = sum(p.numel() for p in self.model.parameters())
        self._load_time_s = time.perf_counter() - load_start

        logger.info(
            "VLMEngine loaded successfully in %.2fs | %s parameters | device: %s",
            self._load_time_s,
            f"{self._param_count:,}",
            self.device,
        )

    def is_loaded(self) -> bool:
        """Return True if the model and processor are currently loaded in memory."""
        return self.model is not None and self.processor is not None

    def unload(self) -> bool:
        """Release the model and processor from memory and trigger garbage collection.

        Returns:
            True if an active model was unloaded, False if already unloaded.
        """
        if not self.is_loaded():
            logger.debug("VLMEngine.unload() called but model is already unloaded.")
            return False

        logger.info("Unloading VLMEngine model and processor...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared after VLM unload.")
            except Exception as e:
                logger.warning("CUDA empty_cache() failed during unload: %s", e)

        logger.info("VLMEngine unloaded successfully.")
        return True

    def _build_messages(self, image: Image.Image, prompt: VLMPrompt) -> List[Dict[str, Any]]:
        """Build the chat-template message list for the processor.

        Args:
            image: PIL RGB Image to analyse.
            prompt: VLMPrompt containing the user instruction.

        Returns:
            List of message dicts in the format expected by the Qwen2.5-VL processor.
        """
        return [
            {
                "role": "system",
                "content": VLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.user_text},
                ],
            },
        ]

    @torch.no_grad()
    def understand(
        self,
        image_input: Union[str, Path, Image.Image],
        mode: PromptMode = PromptMode.GENERAL,
        custom_prompt: Optional[str] = None,
        custom_expected_keys: Optional[List[str]] = None,
        max_new_tokens: int = VLM_MAX_NEW_TOKENS,
        temperature: float = VLM_TEMPERATURE,
        do_sample: bool = VLM_DO_SAMPLE,
    ) -> Dict[str, Any]:
        """Run general image understanding on the provided image.

        Args:
            image_input: Image to analyse. Accepts: PIL Image, file path, URL string, numpy array.
            mode: PromptMode controlling the analysis type. Ignored if custom_prompt is set.
            custom_prompt: Optional custom prompt instruction (overrides mode).
            custom_expected_keys: Expected JSON keys when using a custom prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            do_sample: Whether to use sampling (True) or greedy decoding (False).

        Returns:
            Structured result dictionary with the following guaranteed top-level keys:
                success (bool): Whether inference and parsing both succeeded.
                mode (str): The prompt mode used.
                model_id (str): The model identifier.
                device (str): The device used for inference.
                image_info (dict): Image metadata (size, mode).
                understanding (dict): The VLM-generated structured analysis.
                parse_meta (dict): Parser audit information (recovered, method, warnings).
                timing (dict): Timing breakdown in milliseconds.
                error (str|None): Error message if success is False.
        """
        if not self.is_loaded():
            return {
                "success": False,
                "mode": mode.value if isinstance(mode, PromptMode) else str(mode),
                "model_id": self.model_id,
                "device": self.device,
                "error": "VLMEngine model is not loaded. Call _load_model() or use VLMLifecycleManager.",
                "understanding": {},
                "timing": {},
            }

        overall_start = time.perf_counter()

        # ── Image Loading ─────────────────────────────────────────────────────
        try:
            image = _load_image_input(image_input)
        except (ValueError, FileNotFoundError, IOError) as exc:
            return {
                "success": False,
                "mode": mode.value if isinstance(mode, PromptMode) else str(mode),
                "model_id": self.model_id,
                "device": self.device,
                "error": f"Image loading failed: {exc}",
                "understanding": {},
                "timing": {},
            }

        image = _resize_image_if_needed(image, VLM_MAX_IMAGE_SIZE)
        img_w, img_h = image.size
        image_info = {"width": img_w, "height": img_h, "mode": image.mode}

        # ── Prompt Selection ──────────────────────────────────────────────────
        if custom_prompt:
            prompt = build_custom_prompt(custom_prompt, custom_expected_keys)
        else:
            try:
                prompt = get_prompt(mode)
            except (KeyError, ValueError) as exc:
                return {
                    "success": False,
                    "mode": str(mode),
                    "model_id": self.model_id,
                    "device": self.device,
                    "error": f"Invalid prompt mode: {exc}",
                    "understanding": {},
                    "timing": {},
                }

        # ── Build Chat Messages ───────────────────────────────────────────────
        messages = self._build_messages(image, prompt)

        # ── Tokenize & Preprocess ─────────────────────────────────────────────
        prep_start = time.perf_counter()
        try:
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text_input],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as exc:
            logger.exception("Preprocessing failed for VLM input.")
            return {
                "success": False,
                "mode": prompt.mode.value,
                "model_id": self.model_id,
                "device": self.device,
                "error": f"Preprocessing error: {exc}",
                "image_info": image_info,
                "understanding": {},
                "timing": {},
            }
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # ── Model Inference ───────────────────────────────────────────────────
        inf_start = time.perf_counter()
        try:
            input_ids = inputs.get("input_ids")
            input_length = input_ids.shape[1] if input_ids is not None else 0

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=VLM_MIN_NEW_TOKENS,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

            # Strip input prompt tokens from output
            generated_ids_trimmed = generated_ids[:, input_length:]
        except Exception as exc:
            logger.exception("VLM model.generate() failed.")
            return {
                "success": False,
                "mode": prompt.mode.value,
                "model_id": self.model_id,
                "device": self.device,
                "error": f"Inference error: {exc}",
                "image_info": image_info,
                "understanding": {},
                "timing": {},
            }
        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        # ── Decode Output ─────────────────────────────────────────────────────
        decode_start = time.perf_counter()
        try:
            raw_output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
        except Exception as exc:
            logger.exception("Token decoding failed.")
            return {
                "success": False,
                "mode": prompt.mode.value,
                "model_id": self.model_id,
                "device": self.device,
                "error": f"Decoding error: {exc}",
                "image_info": image_info,
                "understanding": {},
                "timing": {},
            }
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        logger.debug("Raw VLM output (first 500 chars): %s", raw_output[:500])

        # ── Parse Structured Response ─────────────────────────────────────────
        parse_start = time.perf_counter()
        parse_result: ParseResult = parse_vlm_response(raw_output, prompt.expected_keys)
        parse_ms = (time.perf_counter() - parse_start) * 1000.0

        total_ms = (time.perf_counter() - overall_start) * 1000.0

        return {
            "success": parse_result.success,
            "mode": prompt.mode.value,
            "model_id": self.model_id,
            "device": self.device,
            "image_info": image_info,
            "understanding": parse_result.data,
            "parse_meta": {
                "recovered": parse_result.recovered,
                "recovery_method": parse_result.recovery_method,
                "missing_keys": parse_result.missing_keys,
                "extra_keys": parse_result.extra_keys,
                "warnings": parse_result.parse_warnings,
            },
            "timing": {
                "preprocessing_ms": round(prep_ms, 2),
                "inference_ms": round(inf_ms, 2),
                "decoding_ms": round(decode_ms, 2),
                "parsing_ms": round(parse_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "error": None,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return engine metadata and runtime diagnostic information."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "loaded": self.is_loaded(),
            "param_count": self._param_count,
            "load_time_s": round(self._load_time_s, 3),
            "torch_threads": self.torch_threads,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "max_image_size": VLM_MAX_IMAGE_SIZE,
            "max_new_tokens": VLM_MAX_NEW_TOKENS,
        }

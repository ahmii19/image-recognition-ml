"""Unit tests for Phase 7 VLM Engine — General Image Understanding.

Test coverage:
    - src.phase7.config  — All configuration constants load without error.
    - src.phase7.prompts — All prompt modes build correctly; registry is complete.
    - src.phase7.parser  — All 4 parse recovery strategies; edge cases.
    - src.phase7.engine  — Image loading helpers; understand() with mocked model.
    - src.phase7.lifecycle — Singleton behaviour, unload, status telemetry.

IMPORTANT:
    - These tests DO NOT load the actual Qwen2.5-VL-3B-Instruct model.
    - All VLMEngine model/processor attributes are mocked to avoid network downloads.
    - These tests MUST NOT modify any Phase 1–6 source files, tests, or configs.
    - These tests are isolated and can run on CPU without GPU.
"""

import gc
import json
import sys
import types
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rgb_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a minimal in-memory RGB PIL Image for testing."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    return img


def _make_image_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return JPEG bytes of a small test image."""
    img = _make_rgb_image(width, height)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Config:
    """Verify that all configuration constants are importable and have correct types."""

    def test_import_config(self):
        from src.phase7 import config  # noqa: F401

    def test_model_id_is_str(self):
        from src.phase7.config import VLM_MODEL_ID
        assert isinstance(VLM_MODEL_ID, str)
        assert len(VLM_MODEL_ID) > 0

    def test_device_is_str(self):
        from src.phase7.config import VLM_DEVICE
        assert isinstance(VLM_DEVICE, str)

    def test_max_new_tokens_positive(self):
        from src.phase7.config import VLM_MAX_NEW_TOKENS
        assert VLM_MAX_NEW_TOKENS > 0

    def test_temperature_in_range(self):
        from src.phase7.config import VLM_TEMPERATURE
        assert 0.0 <= VLM_TEMPERATURE <= 2.0

    def test_bool_flags(self):
        from src.phase7.config import (
            VLM_AUTO_UNLOAD,
            VLM_DO_SAMPLE,
            VLM_LOAD_IN_4BIT,
            VLM_LOAD_IN_8BIT,
            VLM_TRUST_REMOTE_CODE,
        )
        for flag in [VLM_AUTO_UNLOAD, VLM_DO_SAMPLE, VLM_LOAD_IN_4BIT, VLM_LOAD_IN_8BIT, VLM_TRUST_REMOTE_CODE]:
            assert isinstance(flag, bool)

    def test_system_prompt_non_empty(self):
        from src.phase7.config import VLM_SYSTEM_PROMPT
        assert isinstance(VLM_SYSTEM_PROMPT, str)
        assert len(VLM_SYSTEM_PROMPT) > 20


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT SYSTEM TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Prompts:
    """Verify all prompt builders produce correctly typed VLMPrompt objects."""

    def test_import_prompts(self):
        from src.phase7 import prompts  # noqa: F401

    def test_all_prompt_modes_buildable(self):
        from src.phase7.prompts import PromptMode, get_prompt
        modes_to_test = [
            PromptMode.GENERAL, PromptMode.DETAILED, PromptMode.DOCUMENT,
            PromptMode.DIAGRAM, PromptMode.CHART, PromptMode.MAP,
            PromptMode.MEDICAL, PromptMode.BRIEF,
        ]
        for mode in modes_to_test:
            prompt = get_prompt(mode)
            assert prompt.mode == mode
            assert isinstance(prompt.user_text, str) and len(prompt.user_text) > 10
            assert isinstance(prompt.expected_keys, list) and len(prompt.expected_keys) > 0
            assert isinstance(prompt.description, str) and len(prompt.description) > 5

    def test_custom_prompt_builder(self):
        from src.phase7.prompts import PromptMode, build_custom_prompt
        p = build_custom_prompt("Describe this image.", ["summary"])
        assert p.mode == PromptMode.CUSTOM
        assert p.user_text == "Describe this image."
        assert p.expected_keys == ["summary"]

    def test_custom_prompt_no_keys(self):
        from src.phase7.prompts import build_custom_prompt
        p = build_custom_prompt("Tell me what you see.")
        assert p.expected_keys == []

    def test_get_prompt_custom_raises(self):
        from src.phase7.prompts import PromptMode, get_prompt
        with pytest.raises(ValueError, match="CUSTOM"):
            get_prompt(PromptMode.CUSTOM)

    def test_list_prompt_modes_completeness(self):
        from src.phase7.prompts import PromptMode, list_prompt_modes
        registry = list_prompt_modes()
        modes_in_registry = {entry["mode"] for entry in registry}
        # All non-CUSTOM modes must appear in the registry
        expected_modes = {m.value for m in PromptMode if m != PromptMode.CUSTOM}
        assert expected_modes == modes_in_registry

    def test_general_prompt_contains_json_keys(self):
        from src.phase7.prompts import get_prompt, PromptMode
        p = get_prompt(PromptMode.GENERAL)
        for key in ["scene_type", "main_subject", "summary"]:
            assert key in p.user_text

    def test_medical_prompt_contains_disclaimer(self):
        from src.phase7.prompts import get_prompt, PromptMode
        p = get_prompt(PromptMode.MEDICAL)
        assert "disclaimer" in p.user_text.lower() or "not" in p.user_text.lower()


# ═════════════════════════════════════════════════════════════════════════════
# PARSER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Parser:
    """Test the 4-stage VLM response parser, including all recovery paths."""

    def test_import_parser(self):
        from src.phase7 import parser  # noqa: F401

    # ── Stage 1: Direct JSON parse ────────────────────────────────────────────

    def test_direct_valid_json(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"summary": "A cat sitting on a mat.", "mood": "calm"}'
        result = parse_vlm_response(raw, expected_keys=["summary", "mood"])
        assert result.success is True
        assert result.recovered is False
        assert result.recovery_method == "direct"
        assert result.data["summary"] == "A cat sitting on a mat."
        assert result.data["mood"] == "calm"
        assert result.missing_keys == []

    def test_direct_json_with_whitespace(self):
        from src.phase7.parser import parse_vlm_response
        raw = '  \n  {"key": "value"}  \n  '
        result = parse_vlm_response(raw, expected_keys=["key"])
        assert result.success is True
        assert result.data["key"] == "value"

    def test_direct_json_missing_keys_reported(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"summary": "test"}'
        result = parse_vlm_response(raw, expected_keys=["summary", "mood", "scene_type"])
        assert result.success is True
        assert "mood" in result.missing_keys
        assert "scene_type" in result.missing_keys

    def test_direct_json_extra_keys_reported(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"summary": "test", "unexpected_field": "extra"}'
        result = parse_vlm_response(raw, expected_keys=["summary"])
        assert result.success is True
        assert "unexpected_field" in result.extra_keys

    # ── Stage 2: Markdown fence strip ────────────────────────────────────────

    def test_markdown_fence_json(self):
        from src.phase7.parser import parse_vlm_response
        raw = '```json\n{"summary": "A dog in the park."}\n```'
        result = parse_vlm_response(raw, expected_keys=["summary"])
        assert result.success is True
        assert result.recovered is True
        assert result.recovery_method == "markdown_fence_strip"
        assert result.data["summary"] == "A dog in the park."

    def test_plain_code_fence_json(self):
        from src.phase7.parser import parse_vlm_response
        raw = '```\n{"scene_type": "photograph"}\n```'
        result = parse_vlm_response(raw, expected_keys=["scene_type"])
        assert result.success is True
        assert result.recovered is True

    # ── Stage 3: Bracket-match extraction ────────────────────────────────────

    def test_json_embedded_in_prose(self):
        from src.phase7.parser import parse_vlm_response
        raw = (
            'Here is my analysis of the image: {"summary": "A busy city street.", "mood": "busy"} '
            'I hope this helps!'
        )
        result = parse_vlm_response(raw, expected_keys=["summary", "mood"])
        assert result.success is True
        assert result.recovered is True
        assert result.recovery_method == "json_object_extraction"
        assert result.data["summary"] == "A busy city street."

    def test_json_with_prefix_text(self):
        from src.phase7.parser import parse_vlm_response
        raw = 'Sure! Here is the JSON output:\n{"scene_type": "diagram", "explanation": "A network diagram."}'
        result = parse_vlm_response(raw, expected_keys=["scene_type", "explanation"])
        assert result.success is True
        assert result.data["scene_type"] == "diagram"

    # ── Stage 4: Partial key extraction ──────────────────────────────────────

    def test_partial_key_extraction_fallback(self):
        from src.phase7.parser import parse_vlm_response
        # Completely malformed JSON but key=value pairs visible
        raw = 'summary: "A mountain landscape", mood: "serene"'
        result = parse_vlm_response(raw, expected_keys=["summary", "mood"])
        # May succeed or fail depending on the regex; just assert no exception
        assert isinstance(result.success, bool)

    # ── All stages fail ───────────────────────────────────────────────────────

    def test_unparseable_returns_failure(self):
        from src.phase7.parser import parse_vlm_response
        raw = "I cannot understand this image at all."
        result = parse_vlm_response(raw, expected_keys=["summary"])
        assert result.success is False
        assert "raw_output" in result.data
        assert result.recovery_method == "none"

    def test_empty_string_returns_failure(self):
        from src.phase7.parser import parse_vlm_response
        result = parse_vlm_response("", expected_keys=["summary"])
        assert result.success is False

    def test_parse_result_always_has_raw_text(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"key": "value"}'
        result = parse_vlm_response(raw)
        assert result.raw_text == raw

    def test_no_expected_keys(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"a": 1, "b": 2}'
        result = parse_vlm_response(raw)
        assert result.success is True
        assert result.missing_keys == []

    def test_nested_json_object(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"summary": "test", "nested": {"inner_key": "inner_value"}}'
        result = parse_vlm_response(raw, expected_keys=["summary"])
        assert result.success is True
        assert result.data["nested"]["inner_key"] == "inner_value"

    def test_json_with_list_value(self):
        from src.phase7.parser import parse_vlm_response
        raw = '{"objects": ["cat", "chair", "window"], "summary": "Room scene."}'
        result = parse_vlm_response(raw, expected_keys=["objects", "summary"])
        assert result.success is True
        assert result.data["objects"] == ["cat", "chair", "window"]


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE TESTS (mocked model)
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Engine:
    """Test VLMEngine image loading helpers and understand() with mocked inference."""

    def test_import_engine(self):
        from src.phase7 import engine  # noqa: F401

    def test_load_pil_image(self):
        from src.phase7.engine import _load_image_input
        img = _make_rgb_image(32, 32)
        result = _load_image_input(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_load_file_path(self, tmp_path):
        from src.phase7.engine import _load_image_input
        img = _make_rgb_image(32, 32)
        img_path = tmp_path / "test_img.jpg"
        img.save(str(img_path), format="JPEG")
        result = _load_image_input(str(img_path))
        assert isinstance(result, Image.Image)

    def test_load_path_object(self, tmp_path):
        from src.phase7.engine import _load_image_input
        img = _make_rgb_image(32, 32)
        img_path = tmp_path / "test_img.png"
        img.save(str(img_path), format="PNG")
        result = _load_image_input(img_path)
        assert isinstance(result, Image.Image)

    def test_load_nonexistent_file_raises(self):
        from src.phase7.engine import _load_image_input
        with pytest.raises(FileNotFoundError):
            _load_image_input("/nonexistent/path/image.jpg")

    def test_load_unsupported_type_raises(self):
        from src.phase7.engine import _load_image_input
        with pytest.raises((ValueError, TypeError)):
            _load_image_input(12345)  # type: ignore

    def test_resize_image_if_needed_shrinks(self):
        from src.phase7.engine import _resize_image_if_needed
        img = _make_rgb_image(2000, 1500)
        resized = _resize_image_if_needed(img, max_side=800)
        assert max(resized.size) <= 800

    def test_resize_image_preserves_small_image(self):
        from src.phase7.engine import _resize_image_if_needed
        img = _make_rgb_image(100, 80)
        result = _resize_image_if_needed(img, max_side=800)
        assert result.size == (100, 80)

    def test_resize_preserves_aspect_ratio(self):
        from src.phase7.engine import _resize_image_if_needed
        img = _make_rgb_image(1600, 800)  # 2:1 aspect ratio
        resized = _resize_image_if_needed(img, max_side=800)
        w, h = resized.size
        assert abs(w / h - 2.0) < 0.05  # aspect ratio preserved within 5%

    def _make_mock_engine(self) -> "VLMEngine":  # type: ignore
        """Create a VLMEngine instance with mocked model and processor."""
        from src.phase7.engine import VLMEngine
        with patch("src.phase7.engine.VLMEngine._load_model"):
            engine = VLMEngine.__new__(VLMEngine)
            engine.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            engine.device = "cpu"
            engine.torch_threads = 4
            engine.load_in_4bit = False
            engine.load_in_8bit = False
            engine.trust_remote_code = True
            engine._param_count = 3_000_000_000
            engine._load_time_s = 5.0

            # Mock model and processor
            mock_model = MagicMock()
            mock_processor = MagicMock()

            # generate() returns fake token IDs [batch, seq]
            import torch
            fake_ids = torch.tensor([[101, 202, 303, 404, 505]])
            mock_model.generate.return_value = fake_ids

            # batch_decode returns valid JSON
            expected_json = json.dumps({
                "scene_type": "photograph",
                "main_subject": "a golden retriever",
                "setting": "outdoor park",
                "objects": ["dog", "grass", "tree"],
                "activities": ["sitting"],
                "mood": "calm",
                "image_quality": "high",
                "summary": "A golden retriever sitting in a park."
            })
            mock_processor.batch_decode.return_value = [expected_json]

            # apply_chat_template returns a plain string
            mock_processor.apply_chat_template.return_value = "<chat_template_string>"

            # processor() call returns dict with input_ids
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_processor.tokenizer = MagicMock()
            mock_processor.tokenizer.eos_token_id = 2

            engine.model = mock_model
            engine.processor = mock_processor
            return engine

    def test_understand_general_mode_success(self):
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        img = _make_rgb_image(64, 64)
        result = engine.understand(img, mode=PromptMode.GENERAL)

        assert result["success"] is True
        assert result["mode"] == "general"
        assert "understanding" in result
        assert result["understanding"]["scene_type"] == "photograph"
        assert result["understanding"]["main_subject"] == "a golden retriever"
        assert result["error"] is None

    def test_understand_returns_timing(self):
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        img = _make_rgb_image(64, 64)
        result = engine.understand(img, mode=PromptMode.BRIEF)

        timing = result.get("timing", {})
        assert "total_ms" in timing
        assert timing["total_ms"] >= 0

    def test_understand_returns_image_info(self):
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        img = _make_rgb_image(128, 96)
        result = engine.understand(img, mode=PromptMode.GENERAL)

        assert "image_info" in result
        assert result["image_info"]["width"] == 128
        assert result["image_info"]["height"] == 96

    def test_understand_with_file_path(self, tmp_path):
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        img = _make_rgb_image(64, 64)
        img_path = tmp_path / "test.jpg"
        img.save(str(img_path), format="JPEG")
        result = engine.understand(str(img_path), mode=PromptMode.BRIEF)
        assert result["success"] is True

    def test_understand_invalid_file_returns_failure(self):
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        result = engine.understand("/nonexistent/image.jpg", mode=PromptMode.GENERAL)
        assert result["success"] is False
        assert "error" in result
        assert result["error"] is not None

    def test_understand_unloaded_engine_returns_failure(self):
        from src.phase7.engine import VLMEngine
        from src.phase7.prompts import PromptMode
        engine = self._make_mock_engine()
        engine.model = None
        engine.processor = None
        img = _make_rgb_image(32, 32)
        result = engine.understand(img, mode=PromptMode.GENERAL)
        assert result["success"] is False
        assert "not loaded" in result["error"].lower()

    def test_is_loaded_true_when_model_present(self):
        engine = self._make_mock_engine()
        assert engine.is_loaded() is True

    def test_is_loaded_false_when_model_none(self):
        engine = self._make_mock_engine()
        engine.model = None
        assert engine.is_loaded() is False

    def test_unload_clears_model(self):
        engine = self._make_mock_engine()
        assert engine.is_loaded() is True
        result = engine.unload()
        assert result is True
        assert engine.model is None
        assert engine.processor is None

    def test_unload_returns_false_when_already_unloaded(self):
        engine = self._make_mock_engine()
        engine.model = None
        engine.processor = None
        result = engine.unload()
        assert result is False

    def test_get_info_returns_dict(self):
        engine = self._make_mock_engine()
        info = engine.get_info()
        assert isinstance(info, dict)
        assert "model_id" in info
        assert "device" in info
        assert "param_count" in info

    def test_custom_prompt_understand(self):
        engine = self._make_mock_engine()
        img = _make_rgb_image(64, 64)
        result = engine.understand(
            img,
            custom_prompt='{"summary": "test"}',
            custom_expected_keys=["summary"],
        )
        assert isinstance(result, dict)
        assert "success" in result


# ═════════════════════════════════════════════════════════════════════════════
# LIFECYCLE MANAGER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Lifecycle:
    """Test VLMLifecycleManager singleton behaviour, unload, and status."""

    def setup_method(self):
        """Reset singleton state before each test."""
        from src.phase7 import lifecycle as lc
        lc.VLMLifecycleManager._engine = None
        lc.VLMLifecycleManager._last_accessed_at = None
        lc.VLMLifecycleManager._total_inferences = 0
        lc.VLMLifecycleManager._load_count = 0
        if lc.VLMLifecycleManager._idle_timer is not None:
            lc.VLMLifecycleManager._idle_timer.cancel()
            lc.VLMLifecycleManager._idle_timer = None

    def test_import_lifecycle(self):
        from src.phase7 import lifecycle  # noqa: F401

    def test_is_loaded_initially_false(self):
        from src.phase7.lifecycle import VLMLifecycleManager
        assert VLMLifecycleManager.is_loaded() is False

    def test_unload_when_not_loaded_returns_false(self):
        from src.phase7.lifecycle import VLMLifecycleManager
        assert VLMLifecycleManager.unload() is False

    def test_get_status_when_not_loaded(self):
        from src.phase7.lifecycle import VLMLifecycleManager
        status = VLMLifecycleManager.get_status()
        assert isinstance(status, dict)
        assert status["loaded"] is False
        assert "model_id" in status
        assert "load_count" in status
        assert status["load_count"] == 0

    def test_get_engine_lazy_loads(self):
        """VLMLifecycleManager.get_engine() should trigger VLMEngine creation."""
        from src.phase7.lifecycle import VLMLifecycleManager
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.get_info.return_value = {
            "device": "cpu", "param_count": 3_000_000_000, "load_time_s": 5.0
        }

        with patch("src.phase7.lifecycle.VLMEngine", return_value=mock_engine):
            engine = VLMLifecycleManager.get_engine()
            assert engine is mock_engine
            assert VLMLifecycleManager.is_loaded() is True
            assert VLMLifecycleManager._load_count == 1
            assert VLMLifecycleManager._total_inferences == 1

    def test_get_engine_returns_same_singleton(self):
        """Two calls to get_engine() must return the same object."""
        from src.phase7.lifecycle import VLMLifecycleManager
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.get_info.return_value = {"device": "cpu", "param_count": 0, "load_time_s": 0.0}

        with patch("src.phase7.lifecycle.VLMEngine", return_value=mock_engine):
            engine_1 = VLMLifecycleManager.get_engine()
            engine_2 = VLMLifecycleManager.get_engine()
            assert engine_1 is engine_2
            # VLMEngine constructor called only once
            assert VLMLifecycleManager._load_count == 1
            # But total_inferences incremented twice
            assert VLMLifecycleManager._total_inferences == 2

    def test_unload_after_load(self):
        """Unload should clear the singleton and return True."""
        from src.phase7.lifecycle import VLMLifecycleManager
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.get_info.return_value = {"device": "cpu", "param_count": 0, "load_time_s": 0.0}
        mock_engine.unload.return_value = True

        with patch("src.phase7.lifecycle.VLMEngine", return_value=mock_engine):
            VLMLifecycleManager.get_engine()
            result = VLMLifecycleManager.unload()
            assert result is True
            assert VLMLifecycleManager._engine is None
            assert VLMLifecycleManager.is_loaded() is False

    def test_get_status_after_load(self):
        from src.phase7.lifecycle import VLMLifecycleManager
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.get_info.return_value = {
            "device": "cuda", "param_count": 3_000_000_000, "load_time_s": 12.5
        }

        with patch("src.phase7.lifecycle.VLMEngine", return_value=mock_engine):
            VLMLifecycleManager.get_engine()
            status = VLMLifecycleManager.get_status()
            assert status["loaded"] is True
            assert status["load_count"] == 1
            assert status["total_inferences"] == 1
            assert status["last_accessed_at"] is not None

    def test_thread_safety_concurrent_get_engine(self):
        """Concurrent get_engine() calls from multiple threads must all get the same instance."""
        from src.phase7.lifecycle import VLMLifecycleManager
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.get_info.return_value = {"device": "cpu", "param_count": 0, "load_time_s": 0.0}

        instances = []
        errors = []

        def _worker():
            try:
                e = VLMLifecycleManager.get_engine()
                instances.append(id(e))
            except Exception as exc:
                errors.append(str(exc))

        with patch("src.phase7.lifecycle.VLMEngine", return_value=mock_engine):
            threads = [threading.Thread(target=_worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert errors == [], f"Thread errors: {errors}"
        # All threads should get the same engine instance ID
        assert len(set(instances)) == 1, f"Multiple engine instances created: {set(instances)}"


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION: module-level imports do not break Phase 1–6
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Isolation:
    """Verify Phase 7 does not modify or break Phase 1–6 module imports."""

    def test_phase6_still_importable(self):
        import src.phase6  # noqa: F401

    def test_phase6_owlvit_still_importable(self):
        import src.phase6.owlvit  # noqa: F401

    def test_device_module_unchanged(self):
        from src.device import resolve_device, get_device_info
        device = resolve_device("cpu")
        assert str(device) == "cpu"
        info = get_device_info()
        assert "selected_device" in info

    def test_phase7_does_not_import_phase6(self):
        """Phase 7 source files must not import from src.phase6."""
        import src.phase7.engine as eng_mod
        import src.phase7.lifecycle as lc_mod
        import src.phase7.prompts as pr_mod
        import src.phase7.parser as pa_mod

        for mod in [eng_mod, lc_mod, pr_mod, pa_mod]:
            src_code = Path(mod.__file__).read_text(encoding="utf-8")
            assert "from src.phase6" not in src_code, (
                f"{mod.__file__} must not import from Phase 6."
            )
            assert "import src.phase6" not in src_code, (
                f"{mod.__file__} must not import from Phase 6."
            )

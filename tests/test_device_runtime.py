"""Tests for Centralized Runtime Device Resolution and Hardware Diagnostic Utilities."""

import os
from unittest.mock import patch
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.device import get_device_info, resolve_device
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager


class TestDeviceResolution:
    """Validation suite for resolve_device logic."""

    def test_resolve_device_explicit_cpu(self):
        """Explicit 'cpu' string must return torch.device('cpu')."""
        dev = resolve_device("cpu")
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"

    def test_resolve_device_torch_device_passthrough(self):
        """Passing a torch.device instance directly should return the same instance."""
        cpu_dev = torch.device("cpu")
        resolved = resolve_device(cpu_dev)
        assert resolved == cpu_dev

    def test_resolve_device_auto_on_cpu_host(self):
        """When CUDA is unavailable, 'auto' must gracefully select CPU."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            dev = resolve_device("auto")
            assert dev.type == "cpu"

    def test_resolve_device_auto_on_cuda_host(self):
        """When CUDA is available, 'auto' must select CUDA."""
        with patch.object(torch.cuda, "is_available", return_value=True):
            dev = resolve_device("auto")
            assert dev.type == "cuda"

    def test_resolve_device_explicit_cuda_when_available(self):
        """When CUDA is available, explicit 'cuda' returns torch.device('cuda')."""
        with patch.object(torch.cuda, "is_available", return_value=True):
            dev = resolve_device("cuda")
            assert dev.type == "cuda"

    def test_resolve_device_explicit_cuda_fails_when_unavailable(self):
        """When CUDA is unavailable, explicit 'cuda' must raise a descriptive RuntimeError."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA execution was explicitly requested"):
                resolve_device("cuda")

    def test_resolve_device_invalid_spec(self):
        """Invalid device string must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported device target"):
            resolve_device("invalid_device_xyz")

    def test_resolve_device_from_environment_variable(self):
        """resolve_device(None) must inspect ML_DEVICE environment variable."""
        with patch.dict(os.environ, {"ML_DEVICE": "cpu"}):
            dev = resolve_device()
            assert dev.type == "cpu"

        with patch.dict(os.environ, {"ML_DEVICE": "cuda"}):
            with patch.object(torch.cuda, "is_available", return_value=True):
                dev = resolve_device()
                assert dev.type == "cuda"


class TestDeviceInfoDiagnostics:
    """Validation suite for get_device_info runtime telemetry."""

    def test_get_device_info_structure(self):
        """get_device_info must return complete diagnostic structure."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "selected_device" in info
        assert "cuda_available" in info
        assert "cuda_device_count" in info
        assert "cuda_device_name" in info
        assert "pytorch_version" in info
        assert "cuda_version" in info
        assert "configured_setting" in info
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["pytorch_version"], str)

    def test_get_device_info_with_mocked_cuda(self):
        """get_device_info must reflect active GPU name when CUDA is detected."""
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "device_count", return_value=1):
                with patch.object(torch.cuda, "get_device_name", return_value="NVIDIA Tesla T4"):
                    info = get_device_info()
                    assert info["cuda_available"] is True
                    assert info["cuda_device_count"] == 1
                    assert info["cuda_device_name"] == "NVIDIA Tesla T4"


class TestLifecycleDeviceIntegration:
    """Validation suite for model lifecycle device integration."""

    def test_owlvit_lifecycle_status_device_reporting(self):
        """OWLViTLifecycleManager status must report current resolved device."""
        status = OWLViTLifecycleManager.get_status()
        assert "device" in status
        assert status["device"] in ("cpu", "cuda")


class TestApiModelsDeviceEndpoint:
    """Validation suite for /api/v1/models endpoint device information."""

    @pytest.mark.anyio
    async def test_models_endpoint_device_metadata(self):
        """get_models_info must expose device and device_info."""
        from src.api.routes.models import get_models_info

        data = await get_models_info()
        assert data.open_vocabulary is not None
        assert data.open_vocabulary.device in ("cpu", "cuda")
        assert data.open_vocabulary_detection is not None
        assert data.open_vocabulary_detection.device in ("cpu", "cuda")
        assert data.device_info is not None
        assert data.device_info["selected_device"] in ("cpu", "cuda")
        assert isinstance(data.device_info["cuda_available"], bool)

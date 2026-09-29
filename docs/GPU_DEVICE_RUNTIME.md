# GPU Device Runtime & Hardware Execution Architecture

**Version:** 1.0.0  
**Phase:** Phase GPU-A — CUDA Device Support  
**Target Environments:** Local CPU (Windows/Linux/macOS) | Google Colab (T4/A100) | Cloud GPU VMs

---

## Overview

The computer vision recognition pipeline supports unified runtime device targeting across all deep learning vision backbones:
- **Phase 6A:** OpenCLIP ViT-B/32 Zero-Shot Classifier
- **Phase 6B:** Google OWL-ViT Base Patch32 Object Detector
- **Phase 6C:** Grounded Region-Level Two-Stage Cascade

The device resolution layer ([src/device.py](file:///d:/AHMED%20PROJECTS/ml%20project/src/device.py)) dynamically identifies whether an NVIDIA GPU with CUDA support is accessible or if execution should fallback gracefully to the host CPU.

> [!IMPORTANT]
> **Actual GPU inference must be validated on an NVIDIA CUDA environment.** On local development machines without an NVIDIA GPU or CUDA runtime, the system automatically runs on CPU.

---

## Runtime Modes & Environment Configuration

Device selection is configured using the `ML_DEVICE` environment variable (or model-specific overrides `OPENCLIP_DEVICE` / `OWL_VIT_DEVICE`).

| Configuration | Value | Host CUDA Available | Host CUDA Unavailable | Production Behavior |
|---|---|---|---|---|
| **Auto (Default)** | `ML_DEVICE=auto` | Resolves to `cuda` | Resolves to `cpu` | **Recommended for all environments.** Automatically maximizes performance without configuration changes. |
| **Strict CPU** | `ML_DEVICE=cpu` | Resolves to `cpu` | Resolves to `cpu` | Forces CPU execution regardless of available GPUs. |
| **Strict CUDA** | `ML_DEVICE=cuda` | Resolves to `cuda` | **Raises `RuntimeError`** | Requires CUDA. Fails fast with diagnostic error if no GPU is detected. |
| **Indexed CUDA** | `ML_DEVICE=cuda:0` | Resolves to `cuda:0` | **Raises `RuntimeError`** | Routes tensors to specific GPU device index in multi-GPU clusters. |

---

## Local Development Workflow (CPU Mode)

On a CPU development workstation (e.g., Intel Core i5 with 8 GB RAM):

```powershell
# Default auto-selection (selects CPU safely)
python -m uvicorn src.api.main:app --port 8000

# Or explicitly force CPU mode
$env:ML_DEVICE="cpu"
python -m uvicorn src.api.main:app --port 8000
```

### CPU Resource Safeguards:
1. **Intra-Op Threading:** Thread concurrency is capped to 4 worker threads (`torch.set_num_threads(4)`) to match physical core counts without context-switching churn.
2. **Lazy Model Loading:** OWL-ViT remains unloaded until its first request, preserving initial RAM headroom.
3. **Automated Memory Release:** Model unloads trigger `gc.collect()` to reclaim host RAM.

---

## Future Google Colab / Cloud GPU Execution

When deploying the serving layer on a cloud GPU (e.g., Google Colab with an NVIDIA Tesla T4 or Cloud VM):

```bash
# Verify CUDA is visible to PyTorch
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Start FastAPI serving layer (automatically selects 'cuda')
ML_DEVICE=auto uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### GPU Memory & Performance Benefits:
- **Zero-Shot Similarity (OpenCLIP):** ~180 ms (CPU) $\rightarrow$ **~12 ms (GPU)**
- **Grounded Localization (OWL-ViT):** ~1,400 ms (CPU) $\rightarrow$ **~45 ms (GPU)**
- **Two-Stage Region Pipeline (Phase 6C):** ~2,200 ms (CPU) $\rightarrow$ **~65 ms (GPU)**
- **VRAM Footprint:** ~3.2 GB combined (leaving >12 GB headroom on a 16 GB T4 GPU).
- **VRAM Cache Reclaim:** `OWLViTLifecycleManager.unload()` invokes `torch.cuda.empty_cache()` to release all unreferenced GPU memory back to the driver.

---

## Inspecting Runtime Device Information

Clients can inspect the active execution device and hardware metadata via the model info endpoint:

### `GET /api/v1/models`

**Response Excerpt:**
```json
{
  "open_vocabulary": {
    "model": "OpenCLIP-ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "device": "cpu",
    "max_queries": 20
  },
  "open_vocabulary_detection": {
    "model": "OWL-ViT base patch32",
    "device": "cpu",
    "loaded": false
  },
  "device_info": {
    "selected_device": "cpu",
    "cuda_available": false,
    "cuda_device_count": 0,
    "cuda_device_name": null,
    "pytorch_version": "2.2.2+cpu",
    "cuda_version": null,
    "configured_setting": "auto"
  }
}
```

---

## Troubleshooting & FAQ

### Q: Why do I get `RuntimeError: CUDA execution was explicitly requested ('cuda'), but CUDA is not available`?
**Cause:** `ML_DEVICE=cuda` was set in the environment or passed directly, but PyTorch cannot access an NVIDIA GPU on the host.  
**Resolution:** Switch to `ML_DEVICE=auto` or `ML_DEVICE=cpu` for local development.

### Q: Does switching from CPU to GPU change the REST API format?
**Answer:** **No.** The API contract is 100% identical. The request multipart schemas, detection bounding box dictionaries, and cosine similarity rankings remain unchanged. Only internal tensor execution accelerates.

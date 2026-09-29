# PHASE 6B-A PRE-BENCHMARK FORENSIC AUDIT
**Date:** September 25, 2026  
**Scope:** Isolated Benchmark Evaluation of OWL-ViT (`google/owlvit-base-patch32`)  
**Status:** AUDIT COMPLETE — ISOLATION PLAN ESTABLISHED  

---

## 1. Executive Environment Baseline

| Parameter | Current System State | Target Constraint | Status |
|---|---|---|:---:|
| **Operating System** | Windows 10/11 Pro AMD64 (Build 26200) | Windows 64-bit | **CONFIRMED** |
| **CPU Architecture** | Intel Core i5-8265U (4 Physical / 8 Logical Cores) | Intel i5 Quad-Core | **CONFIRMED** |
| **Usable Physical RAM** | 7.38 GB (Available: 2.36 GB – 3.10 GB dynamic) | 8.0 GB Host | **CONFIRMED** |
| **Disk Space (D:)** | 39.77 GB Free (Total: 96.88 GB) | $> 5.0\text{ GB}$ Free | **CONFIRMED** |
| **GPU / Accelerator** | Intel UHD Graphics 620 (No NVIDIA GPU) | CPU-Only (No CUDA) | **CONFIRMED** |
| **PyTorch CUDA Status** | `torch.cuda.is_available() == False` | CUDA Unavailable | **CONFIRMED** |
| **PyTorch Threading** | `torch.get_num_threads() == 4` | 4 Worker Threads | **CONFIRMED** |

---

## 2. Dependency Audit & Integrity Matrix

| Package | Installed Version | Production Pin | Benchmark Required Action |
|---|---|---|---|
| `python` | `3.11.16` | Python 3.11+ | **PRESERVE** |
| `tensorflow` | `2.17.1` | `tensorflow==2.17.1` | **PRESERVE UNCHANGED** |
| `keras` | `3.15.1` | `keras==3.15.1` | **PRESERVE UNCHANGED** |
| `numpy` | `1.26.4` | `numpy==1.26.4` | **PRESERVE UNCHANGED** |
| `torch` | `2.14.0+cpu` | PyTorch CPU-only | **PRESERVE UNCHANGED** |
| `torchvision` | `0.29.0+cpu` | Torchvision CPU | **PRESERVE UNCHANGED** |
| `open_clip_torch`| `3.3.0` | OpenCLIP Phase 6A | **PRESERVE UNCHANGED** |
| `fastapi` | `0.141.1` | FastAPI Phase 4 | **PRESERVE UNCHANGED** |
| `uvicorn` | `0.53.0` | ASGI Server | **PRESERVE UNCHANGED** |
| `pydantic` | `2.13.5` | Data Validation | **PRESERVE UNCHANGED** |
| `pillow` | `12.3.0` | Image Processing | **PRESERVE UNCHANGED** |
| `huggingface_hub`| `2.0.0` | Model Download/Cache | **PRESERVE** |
| **`transformers`** | **NOT INSTALLED** | *Not in requirements.txt* | **TEMPORARY BENCHMARK INSTALL ONLY** |

---

## 3. Benchmark Model Identity

- **Candidate Model:** OWL-ViT (Open-Vocabulary Object Detector)
- **Architecture:** `OwlViTForObjectDetection` (ViT-B/32 backbone + cross-attention multimodal detection heads)
- **Pretrained Checkpoint ID:** `google/owlvit-base-patch32`
- **Processor:** `OwlViTProcessor`
- **Tokenizer:** CLIPTokenizer (`transformers`)
- **Estimated Checkpoint Size:** $\sim 600\text{ MB}$ weights ($\sim 155\text{M}$ parameters)
- **Cache Location:** `~/.cache/huggingface/hub/models--google--owlvit-base-patch32`

---

## 4. Preservation & Isolation Boundary

1. **`requirements.txt`:** MUST NOT be modified. Production requirements remain pinned to existing Phase 1–6A baseline.
2. **Production Code:** No changes to `src/phase1/`, `src/phase2/`, `src/phase3/`, `src/phase4/`, `src/phase5/`, `src/phase6/`, or `frontend/`.
3. **Benchmark Directory:** All benchmark scripts, test images, execution logs, and visualizations MUST live strictly in:
   `scratch/phase6b_owlvit_benchmark/`
4. **Temporary Dependency Installation:** Install `transformers` (`pip install transformers --no-deps` or minimal compatible installation) without touching locked core versions (`numpy==1.26.4`, `torch==2.14.0+cpu`, `tensorflow==2.17.1`).

---

## 5. Pre-Audit Sign-Off

The system is ready for isolated execution of the Phase 6B-A benchmark in `scratch/phase6b_owlvit_benchmark/`.

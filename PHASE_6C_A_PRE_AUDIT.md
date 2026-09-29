# Phase 6C-A Pre-Benchmark Forensic Audit: Isolated Region-Level Open-Vocabulary Recognition

**Date**: September 25, 2026  
**Phase**: Phase 6C-A — Isolated Region-Level Open-Vocabulary Recognition Benchmark  
**Role**: Senior Computer-Vision Researcher & ML Systems Performance Engineer  
**Objective**: Read-only pre-benchmark audit and experimental design to evaluate whether grounding OpenCLIP on OWL-ViT bounding-box crops provides measurable accuracy and ambiguity-resolution improvements over whole-image recognition on an 8 GB RAM CPU-only system.

---

## 1. System Environment & Hardware Baseline

| Resource | Specification | Operational Constraint |
| :--- | :--- | :--- |
| **Host CPU** | Intel Core i5-8265U (4 Cores / 8 Threads) | Pinned to 4 intra-op CPU threads (`torch.set_num_threads(4)`) |
| **System RAM** | 8.00 GB Physical (~7.38 GB usable) | Host working set must stay within bounded limit to avoid paging/OOM |
| **Acceleration** | Intel UHD Graphics 620 | CPU-only (CUDA not available) |
| **Python Environment** | Python 3.11.16 | `.venv` |
| **Frameworks** | PyTorch 2.5.1+cpu, Transformers 4.40+, OpenCLIP 3.0+ | Pinned in environment |
| **Model Caches** | `~/.cache/huggingface/hub/` | `google/owlvit-base-patch32` (586.1 MB), `laion/CLIP-ViT-B-32` (605.2 MB) |

---

## 2. Reusable Codebase Subsystems & Assets

1. **OWL-ViT Open-Vocabulary Detector**:
   - Class: `src/phase6/owlvit/detector.py` (`OWLViTDetector`)
   - Lifecycle: `src/phase6/owlvit/lifecycle.py` (`OWLViTLifecycleManager`)
   - Execution: ViT-B/32 backbone + cross-attention multi-modal projection heads (153.2M params).
2. **OpenCLIP Zero-Shot Semantic Classifier**:
   - Class: `src/phase6/open_vocab_classifier.py` (`OpenCLIPClassifier`)
   - Backbone: `ViT-B/32` (`laion2b_s34b_b79k`, 512-dim embedding, 151.3M params).
3. **Image Validation**:
   - Module: `src/phase3/image_validator.py` (`validate_and_load_image`).
4. **Validation Datasets**:
   - Diverse real-world images in `scratch/phase6b_owlvit_benchmark/images/` and `data/phase2/dataset_cache/flower_photos/`.

---

## 3. Potential Memory Conflicts & Mitigation Strategy

Running **both** OWL-ViT (~1,012 MB resident) and OpenCLIP (~550 MB resident) simultaneously produces a combined working set of **~1.56 GB–1.75 GB**.

### Safeguards:
1. **Sequential Model Loading / Batching**: Maintain single singleton instances of each model; avoid redundant re-instantiations.
2. **Batch Image Encoding**: When evaluating multiple detected regions, pass crops as a stacked tensor `(B, 3, 224, 224)` into `open_clip.model.encode_image(batch)` to maximize CPU vector throughput without loop overhead.
3. **Explicit Memory Profiling**: Measure process Working Set via Windows `KERNEL32.GetProcessMemoryInfo` and system free RAM via `GlobalMemoryStatusEx`.
4. **Garbage Collection**: Run `gc.collect()` between benchmark suites.

---

## 4. Benchmark Experimental Design

```
                     Input Image (Full Resolution)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [Baseline Evaluation]               [Two-Stage Region Pipeline]
      Whole-Image OpenCLIP                        OWL-ViT Detector
      - Zero-shot cosine ranking                  - Query-based box detection
      - Top-1 & Top-5 accuracy                    - Bounding box validation
                                                   │
                                                   ▼
                                          Crop Object Regions
                                          - Padding: 0%, 5%, 10%, 20%
                                          - Aspect ratio preservation
                                                   │
                                                   ▼
                                          OpenCLIP Region Analysis
                                          - Refined semantic candidates
                                          - Ambiguity pair testing
                                          - Fine-grained attributes
                                                   │
                                                   ▼
                                         Comparison & Analytics
                                         - Accuracy delta (Whole vs Region)
                                         - Ambiguity resolution rate
                                         - CPU Latency & Memory profiling
```

---

## 5. Planned Benchmark Directory & File Structure

```
scratch/phase6c_a_region_benchmark/
├── benchmark.py                  # Main execution harness
├── region_pipeline.py            # Region cropping, padding, and OpenCLIP evaluation engine
├── evaluation.py                 # Ground truth metrics, ambiguity tests, attribute analytics
├── dataset_manifest.py           # Pre-registered 18+ image dataset with ground-truth labels
├── images/                       # Real-world benchmark images
├── results/                      # Raw JSON benchmark outputs
├── visualizations/               # Side-by-side comparison images with bounding boxes & labels
└── PHASE_6C_A_REPORT.md          # Comprehensive benchmark results and feasibility analysis
```

---

## 6. Pre-Benchmark Integrity Verification

- [x] Read-only forensic audit completed.
- [x] Strict isolation guaranteed under `scratch/phase6c_a_region_benchmark/`.
- [x] Zero modifications to `src/`, production routes, frontend, or dependencies.
- [x] Pre-defined ground truth annotations established to prevent post-hoc evaluation bias.

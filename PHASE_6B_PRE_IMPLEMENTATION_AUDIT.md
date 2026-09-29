# Phase 6B Pre-Implementation Forensic Audit: Production Open-Vocabulary Object Detection (OWL-ViT)

**Date**: September 25, 2026  
**Phase**: Phase 6B — Production Open-Vocabulary Object Detection  
**Role**: Senior ML Systems Engineer  
**Objective**: Architectural audit and pre-implementation blueprint for integrating OWL-ViT base patch32 (`google/owlvit-base-patch32`) into the FastAPI serving layer and Next.js web application with strict lazy loading and memory-aware lifecycle management.

---

## 1. Executive Summary & Context

The project currently contains verified, fully passing implementations across Phases 1 through 6A:
- **Phase 1**: Foundational CNN (CIFAR-10)
- **Phase 2**: Transfer Learning (5-class flower classification with EfficientNetB0)
- **Phase 3**: General ImageNet-1K Classification (MobileNetV2) + COCO Object Detection (SSD-MobileNetV2, 80 categories)
- **Phase 4**: FastAPI Serving Layer (`/api/v1/health`, `/api/v1/ready`, `/api/v1/models`, `/api/v1/recognize`, `/api/v1/classify`, `/api/v1/detect`)
- **Phase 5**: Interactive Next.js Web UI with responsive synchronized bounding box canvas
- **Phase 6A**: OpenCLIP ViT-B/32 image-level zero-shot semantic recognition (`/api/v1/open-vocabulary`)
- **Phase 6B-A**: Isolated OWL-ViT Base Patch32 benchmark (measured ~1.01 GB peak RAM, 1,281 ms warm 1-query latency, 586.1 MB cache footprint, and confirmed multi-object localization capabilities).

The purpose of Phase 6B is to promote OWL-ViT from the isolated benchmark to a first-class production capability without destabilizing the host system (Intel Core i5-8265U, 8 GB RAM, CPU only) or regressing any existing capabilities.

---

## 2. Existing Subsystems Audit

### 2.1 Phase 3 ML Engine & Image Validation
- **Engine**: `src/phase3/engine.py` orchestrates `ImageNetClassifier` (`MobileNetV2`) and `COCOObjectDetector` (`SSD-MobileNetV2`).
- **Image Validation**: `src/phase3/image_validator.py` (`validate_and_load_image`) provides rigorous multi-layer validation (magic bytes, dimensions, aspect ratios, PIL verification, channel normalization).
- **Preservation Strategy**: The Phase 6B detector will reuse `validate_and_load_image` directly without creating redundant image processing pipelines.

### 2.2 Phase 4 FastAPI Serving Architecture & Lifecycle
- **App Lifecycle**: `src/api/main.py` uses `lifespan(app: FastAPI)` context manager.
- **Current App State**:
  - `app.state.engine` -> `AdvancedRecognitionEngine(lazy_load=False)`
  - `app.state.open_vocab_engine` -> `OpenVocabEngine(lazy_load=False)`
- **Dependencies**: `src/api/dependencies.py` provides `get_recognition_engine`, `get_open_vocab_engine`, and `validate_and_read_image_upload`.
- **Preservation Strategy**:
  - **CRITICAL**: OWL-ViT MUST NOT load during `lifespan` startup.
  - A new lazy dependency `get_owlvit_detector` will access a thread-safe singleton that initializes on first request.

### 2.3 Phase 6A OpenCLIP Architecture
- **Classifier**: `src/phase6/open_vocab_classifier.py` (`OpenVocabClassifier`) wraps `open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")`.
- **Engine**: `src/phase6/engine.py` (`OpenVocabEngine`) provides query validation, prompt formatting, and inference timing.
- **Route**: `src/api/routes/open_vocab.py` exposes `POST /api/v1/open-vocabulary`.
- **Preservation Strategy**: OpenCLIP remains untouched. OWL-ViT will be modularized in a distinct package `src/phase6/owlvit/` with its own configuration, lifecycle manager, detector, and schemas.

### 2.4 Next.js Frontend Architecture
- **API Client**: `frontend/src/lib/api.ts` provides typed wrappers (`recognizeImage`, `classifyImage`, `detectImage`, `recognizeOpenVocab`, `getModelsInfo`, `checkHealth`).
- **UI Components**:
  - `frontend/src/app/page.tsx`: Mode selector, image upload, results dispatcher.
  - `frontend/src/components/ImagePreview.tsx`: Mode selector tiles (`all`, `classification`, `detection`, `open_vocabulary`).
  - `frontend/src/components/DetectionViewer.tsx`: Responsive bounding box canvas overlay with synchronized card hover/selection.
  - `frontend/src/components/OpenVocabPanel.tsx`: Query input chips and similarity score ranking.
- **Integration Strategy**: Add a new mode `"open_vocabulary_detection"`, dedicated UI controls (presets, prompt templates, threshold sliders), and adapt `DetectionViewer` to render open-vocabulary bounding boxes with detection confidence scores.

---

## 3. Host Constraints & Memory Conflict Analysis

| Parameter | Host Specification | Operational Boundary |
| :--- | :--- | :--- |
| **CPU** | Intel Core i5-8265U (4C/8T) | PyTorch threads pinned to 4 cores |
| **Physical RAM** | 8.00 GB (~7.38 GB usable) | Host currently has ~1.8–3.0 GB free |
| **GPU / Acceleration** | Intel UHD Graphics 620 | CPU-only (no CUDA, no TensorRT) |
| **Phase 1–3 Resident RAM** | ~650 MB (TF/Keras MobileNetV2 + SSD) | Always resident in memory |
| **Phase 6A Resident RAM** | ~550 MB (OpenCLIP ViT-B/32) | Always resident in memory |
| **OWL-ViT Resident RAM** | ~1,012 MB (Vision + Text + Cross-Attn) | **LAZY LOADED ON DEMAND** |
| **Combined Worst-Case RAM** | **~2.21 GB** Python Working Set | Leaves >1.0 GB host OS headroom |

### Memory Conflict Mitigation Strategy:
1. **Lazy Loading**: `OWLViTDetector` singleton is instantiated **only** when `/api/v1/open-vocabulary/detect` is invoked.
2. **Thread-Safe Singleton**: A `threading.Lock` protects the loading block to prevent race conditions from concurrent requests allocating multiple ~1 GB model copies.
3. **Configurable Memory Policy**: Environment variables (`OWL_VIT_AUTO_UNLOAD`, `OWL_VIT_IDLE_UNLOAD_SECONDS`) enable automated idle resource reclamation via Python `gc.collect()` and PyTorch memory cleanup.
4. **Controlled Concurrency**: An async semaphore (`asyncio.Semaphore(1)`) guards CPU-heavy OWL-ViT forward passes to prevent CPU saturation and latency degradation.

---

## 4. Architectural Blueprint

```
                      +-----------------------------+
                      |   HTTP Request (Multipart)  |
                      |   Image + Queries + Params  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  FastAPI Routing & Security |
                      |  - 10MB Stream Validation   |
                      |  - Phase 3 Image Validator  |
                      |  - Query Sanitizer & Prompt |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  OWLViTLifecycleManager     |
                      |  - Thread-Safe Singleton    |
                      |  - Lazy Model Instantiation |
                      |  - Idle Timer / Auto-Unload |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  OWLViTDetector (CPU-4T)    |
                      |  - google/owlvit-base-patch32|
                      |  - Image & Text Cross-Attn  |
                      |  - Box Regression & NMS     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Structured Output Schema   |
                      |  - Image Meta + Timing      |
                      |  - Bounding Boxes (px & norm)|
                      |  - Detection IDs (Phase 6C) |
                      +-----------------------------+
```

---

## 5. File Modification & Creation Inventory

### 5.1 New Files to Create:
1. `src/phase6/owlvit/__init__.py`: Package exports for OWL-ViT components.
2. `src/phase6/owlvit/config.py`: Environment variables and default parameters (`OWL_VIT_MODEL_ID`, `OWL_VIT_TORCH_THREADS`, `DEFAULT_PROMPT_TEMPLATE`, `MAX_TEXT_QUERIES`, idle timeouts).
3. `src/phase6/owlvit/detector.py`: Core `OWLViTDetector` performing model inference, box unscaling, and query matching.
4. `src/phase6/owlvit/lifecycle.py`: Thread-safe singleton `OWLViTLifecycleManager` with lazy loading, concurrency locks, and memory unload management.
5. `src/api/routes/open_vocab_detection.py`: Dedicated FastAPI router for `POST /api/v1/open-vocabulary/detect` and `POST /api/v1/open-vocabulary/unload`.
6. `frontend/src/components/OpenVocabDetectionPanel.tsx`: Next.js UI component for query chips, presets, prompt templates, and detection threshold sliders.
7. `tests/test_phase6b_owlvit.py`: Unit tests for OWL-ViT config, prompt normalization, lazy loading, inference, and error handling.
8. `tests/test_api_owlvit.py`: Integration tests for `/api/v1/open-vocabulary/detect` and `/api/v1/models` runtime state.

### 5.2 Existing Files to Modify:
1. `requirements.txt`: Add `transformers>=4.40.0` for production dependency alignment.
2. `src/api/schemas.py`: Add `OpenVocabDetectionResponse`, `OpenVocabDetectionModelInfo`, `OWLViTBoxCoordinates`, `OWLViTDetectionItem`, `OWLViTQueryGroup`.
3. `src/api/dependencies.py`: Add `get_owlvit_detector` dependency.
4. `src/api/routes/models.py`: Extend `GET /api/v1/models` to include `open_vocabulary_detection` metadata with dynamic `loaded: bool` status.
5. `src/api/main.py`: Include `open_vocab_detection.router` and ensure clean shutdown hooks without loading OWL-ViT at startup.
6. `frontend/src/types/api.ts`: Add TypeScript interfaces for Open-Vocabulary Detection responses and models.
7. `frontend/src/lib/api.ts`: Add `detectOpenVocab` and `unloadOpenVocabDetector` client functions.
8. `frontend/src/components/ImagePreview.tsx`: Add `"open_vocabulary_detection"` mode selector card.
9. `frontend/src/app/page.tsx`: Integrate Open-Vocabulary Detection workflow and bounding box visualization.

---

## 6. Pre-Implementation Verification Checklist

- [x] Pre-audit complete and documented
- [x] Zero production regressions planned
- [x] CPU-only execution confirmed (4 threads, no CUDA)
- [x] Lazy loading strictly designed (no loading at app startup)
- [x] Thread-safe singleton lock planned
- [x] Prompt ensembling (`"a photo of a {}"`) included
- [x] Phase 6C data contract preparation included (pixel + normalized coordinates, detection ID)
- [x] Full regression test suites identified

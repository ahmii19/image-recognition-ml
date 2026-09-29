# Phase 6C Pre-Implementation Forensic Audit: Production Region-Level Open-Vocabulary Recognition

**Date**: September 28, 2026  
**Investigator**: Senior Computer-Vision Engineer & Production ML Systems Architect  
**Objective**: Comprehensive read-only forensic audit and architectural specification for integrating Phase 6C (Two-Stage Grounded Region-Level Open-Vocabulary Recognition Cascade) into the production serving system.

---

## 1. Current Architecture Review

The current production serving system consists of:
- **Phase 1 & 2**: Foundational CNN & Transfer Learning (MobileNetV2, CIFAR-10 & 5-Flower dataset).
- **Phase 3**: Advanced Recognition Engine (MobileNetV2 1,000 ImageNet + SSD-MobileNetV2 80 COCO).
- **Phase 4**: FastAPI Serving Layer with strict validation, streaming byte limits, and lifecycle management.
- **Phase 5**: Next.js 14 Web UI with interactive bounding box visualization (`DetectionViewer`), mode selector, and telemetry drawers.
- **Phase 6A**: OpenCLIP ViT-B/32 Zero-Shot Image-Level Classifier (`/api/v1/open-vocabulary`).
- **Phase 6B**: Google OWL-ViT Base Patch32 Open-Vocabulary Object Detector with lazy loading & unload controls (`/api/v1/open-vocabulary/detect`).

---

## 2. Reusable Subsystems & Extension Points

1. **Object Detection Stage (Stage 1)**:
   - `src/phase6/owlvit/detector.py` (`OWLViTDetector`): Handles image validation, candidate query sanitization, and grounded box predictions.
   - `src/phase6/owlvit/lifecycle.py` (`OWLViTLifecycleManager`): Handles thread-safe singleton instantiation, lazy loading on first request, and memory reclamation via `unload()`.
2. **Semantic Recognition Stage (Stage 2)**:
   - `src/phase6/open_vocab_classifier.py` (`OpenCLIPClassifier`): Pre-loaded ViT-B/32 backbone with `model.encode_image()` and `model.encode_text()`.
   - Tokenizer and transform functions are pre-compiled and thread-safe.
3. **API Serving Infrastructure**:
   - `src/api/dependencies.py`: `get_owlvit_detector()` and `get_open_vocab_engine()`.
   - `src/api/errors.py`: Custom HTTP exceptions with machine-readable error codes.
4. **Frontend Architecture**:
   - `frontend/src/components/DetectionViewer.tsx`: Interactive SVG/Canvas bounding box overlay with hover/selection synchronization.
   - `frontend/src/lib/api.ts` & `frontend/src/types/api.ts`: Centralized client and type definitions.

---

## 3. Production Extension Points & Proposed Changes

### 3.1 Backend Modules to Create:
1. `src/phase6/region_pipeline.py`:
   - `RegionRecognitionPipeline`: Encapsulates 10% symmetrical context crop extraction, boundary clamping, batch preparation, single-pass batch image encoding, single-pass text embedding reuse, and vectorized cosine similarity ranking.
2. `tests/test_phase6c_region.py`:
   - Unit tests covering crop padding, coordinate validation, zero-area rejection, region limits, batch tensor execution, text embedding reuse, and ranking.
3. `tests/test_api_region_recognition.py`:
   - API integration tests for `POST /api/v1/open-vocabulary/region-recognition`, parameter validation, and error contracts.

### 3.2 Backend Modules to Modify:
1. `src/phase6/config.py`:
   - Add `DEFAULT_CROP_PADDING_PERCENT = 10`, `MAX_REGIONS = 20`, `DEFAULT_MAX_REGIONS = 20`, `OPENCLIP_REGION_BATCH_SIZE = 20`.
2. `src/api/schemas.py`:
   - Define `RegionCropBounds`, `RegionDetectionMeta`, `RegionCandidateRank`, `RegionRecognitionItem`, `RegionPipelineSummary`, `RegionPipelineTiming`, `RegionRecognitionResponse`, and `RegionIntelligenceModelInfo`.
3. `src/api/routes/open_vocab_detection.py` (or dedicated `src/api/routes/region_recognition.py`):
   - Expose `POST /api/v1/open-vocabulary/region-recognition`.
4. `src/api/routes/models.py`:
   - Extend `GET /api/v1/models` to include `region_intelligence` capability metadata and add `"region_recognition"` to `supported_modes`.
5. `src/api/main.py`:
   - Register route and ensure graceful cleanup on shutdown.

### 3.3 Frontend Modules to Create / Modify:
1. `frontend/src/types/api.ts`:
   - Add TypeScript interfaces for region intelligence responses, crops, and update `RecognitionMode` union type to include `"region_recognition"`.
2. `frontend/src/lib/api.ts`:
   - Add `recognizeRegions()` client function.
3. `frontend/src/components/RegionIntelligencePanel.tsx`:
   - Create interactive control and results breakdown component displaying detected regions, OWL-ViT detection scores, OpenCLIP refinements, similarity scores, and crop specifications.
4. `frontend/src/components/ImagePreview.tsx`:
   - Add 6th mode selector button: **"Region Intelligence"** (`region_recognition`).
5. `frontend/src/app/page.tsx`:
   - Integrate `"region_recognition"` mode into state, execution handlers, and synchronized bounding box views.
6. `frontend/src/__tests__/region_intelligence.test.mjs`:
   - Frontend unit tests for region data mapping and coordinate math.

---

## 4. Concurrency, Memory & Performance Safeguards

1. **Batch Image Encoding (Mandatory)**:
   - For $N$ detected regions ($N \le 20$), crop all bounding boxes with 10% context padding, preprocess each crop with `clip.preprocess`, stack into a single tensor `(N, 3, 224, 224)`, and execute a single forward pass: `image_features = clip.model.encode_image(batch_tensor)`.
2. **Text Embedding Reuse (Mandatory)**:
   - Candidate text queries are tokenized and encoded **once per request** (`(Q, D)`). Cosine similarities across all $N$ regions are computed in a single vectorized matrix multiplication:
     $$\mathbf{S} = \mathbf{F}_{\text{image}} \times \mathbf{F}_{\text{text}}^T \quad (N \times D) \times (D \times Q) \to (N \times Q)$$
3. **Singleton Pattern & Memory Boundaries**:
   - Reuses pre-loaded `OpenCLIPClassifier` and lazy-loaded `OWLViTDetector`. No new model instances are constructed.
   - Peak combined resident memory: **~1,562 MB**, leaving **>1.2 GB of free physical RAM** on the 8 GB machine.
4. **Concurrency Protection**:
   - Shared `asyncio.Semaphore(1)` ensures only one CPU-heavy cascade executes at any instant, preventing memory spikes or thread starvation.

---

## 5. Backward Compatibility Guarantee

- All existing endpoints (`/api/v1/health`, `/api/v1/ready`, `/api/v1/models`, `/api/v1/recognize`, `/api/v1/classify`, `/api/v1/detect`, `/api/v1/open-vocabulary`, `/api/v1/open-vocabulary/detect`) remain untouched and 100% backward compatible.
- All 72 existing backend tests and 10 frontend tests must pass with 0 regressions.

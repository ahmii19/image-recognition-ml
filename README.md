# CIFAR-10 Image Recognition ML Pipeline

An end-to-end, modular, and educational Machine Learning project built from scratch with **TensorFlow / Keras** for image classification on the **CIFAR-10** dataset.

---

## 1. Project Overview

This project implements a complete, self-contained Convolutional Neural Network (CNN) pipeline without relying on external cloud APIs or pre-trained models. The entire ML lifecycle is implemented in clean, modular Python components:

```
Input Image (e.g., dog, airplane, ship)
  │
  ▼
Preprocessing & Normalization (32x32x3, float32 ∈ [0, 1])
  │
  ▼
Convolutional Neural Network (3 Conv Blocks + Dense Head)
  │
  ▼
Softmax Probability Distribution (10 classes)
  │
  ▼
Output: Predicted Class (e.g., "Dog") + Confidence (e.g., 94.72%) + Top-K Rankings
```

---

## 2. Machine Learning Fundamentals & Core Concepts

### Dataset Splitting Strategy
- **Training Set (40,000 images)**: Used exclusively to compute loss and update model weights via backpropagation.
- **Validation Set (10,000 images)**: Used during training to monitor generalization performance, tune hyperparameters, and trigger callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
- **Test Set (10,000 images)**: Held-out, untouched evaluation set used *only once* after training is complete to obtain unbiased performance metrics.

### CNN & Deep Learning Concepts Explained
- **Convolution ($Conv2D$)**: Applies learnable spatial filter kernels (e.g., $3 \times 3$) across the image. It computes dot products across local receptive fields to detect spatial patterns like edges, textures, and object parts regardless of their position in the frame.
- **Filters & Feature Maps**: A filter is a small weight matrix ($3 \times 3 \times C$). Convolving $K$ filters over an input generates a 3D volume of $K$ distinct *feature maps*, each highlighting specific visual motifs.
- **Pooling ($MaxPooling2D$)**: Downsamples feature maps by taking the maximum value in local $2 \times 2$ windows. This halves spatial dimensions ($32 \to 16 \to 8 \to 4$), reduces computational parameters, and provides translational invariance.
- **Batch Normalization**: Re-centers and re-scales intermediate layer activations across mini-batches. It mitigates internal covariate shift, accelerates training convergence, and acts as a mild regularizer.
- **Activation Functions (ReLU & Softmax)**:
  - **ReLU ($\text{max}(0, x)$)** introduces non-linearity without suffering from vanishing gradients for positive inputs.
  - **Softmax ($\frac{e^{z_i}}{\sum_j e^{z_j}}$)** transforms raw logits into a normalized probability distribution where $\sum p_i = 1.0$.
- **Loss Function (Cross-Entropy)**:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
  Measures the divergence between the true one-hot distribution $y$ and the predicted probability distribution $\hat{y}$.
- **Regularization Techniques**:
  - **Data Augmentation**: Generates synthetic variations (random horizontal flips, subtle rotations, translations, zooms) to prevent the CNN from memorizing exact pixel layouts.
  - **Dropout**: Randomly deactivates a fraction (e.g., $20\%-50\%$) of neurons during forward passes in training, forcing the network to learn robust, redundant representations.
  - **Early Stopping**: Halts training when validation loss stops improving, preventing over-training on the training partition.

---

## 3. Dataset Characteristics (CIFAR-10)

| Attribute | Specification |
| :--- | :--- |
| **Total Images** | 60,000 color images |
| **Training Pool** | 50,000 images (partitioned into 40k train / 10k val) |
| **Test Set** | 10,000 images (official standard benchmark) |
| **Resolution** | $32 \times 32$ pixels |
| **Color Channels** | 3 (RGB) |
| **Number of Classes**| 10 mutually exclusive categories |
| **Classes** | `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` |

---

## 4. CNN Architecture

```
Layer (type)                     Output Shape          Param #     Description
=========================================================================================
InputLayer                       (None, 32, 32, 3)     0           Raw RGB Image (float32 [0,1])
-----------------------------------------------------------------------------------------
Conv2D (32 filters, 3x3, same)   (None, 32, 32, 32)    896         Low-level feature extraction
BatchNormalization               (None, 32, 32, 32)    128         Activation stabilization
Conv2D (32 filters, 3x3, same)   (None, 32, 32, 32)    9,248       Refined low-level edges
BatchNormalization               (None, 32, 32, 32)    128         Activation stabilization
MaxPooling2D (2x2)               (None, 16, 16, 32)    0           Spatial downsampling
Dropout (0.20)                   (None, 16, 16, 32)    0           Regularization
-----------------------------------------------------------------------------------------
Conv2D (64 filters, 3x3, same)   (None, 16, 16, 64)    18,496      Mid-level textures & patterns
BatchNormalization               (None, 16, 16, 64)    256         Activation stabilization
Conv2D (64 filters, 3x3, same)   (None, 16, 16, 64)    36,928      Combined texture patterns
BatchNormalization               (None, 16, 16, 64)    256         Activation stabilization
MaxPooling2D (2x2)               (None, 8, 8, 64)      0           Spatial downsampling
Dropout (0.30)                   (None, 8, 8, 64)      0           Regularization
-----------------------------------------------------------------------------------------
Conv2D (128 filters, 3x3, same)  (None, 8, 8, 128)     73,856      High-level semantic parts
BatchNormalization               (None, 8, 8, 128)     512         Activation stabilization
Conv2D (128 filters, 3x3, same)  (None, 8, 8, 128)     147,584     Complex object geometry
BatchNormalization               (None, 8, 8, 128)     512         Activation stabilization
MaxPooling2D (2x2)               (None, 4, 4, 128)     0           Spatial downsampling
Dropout (0.40)                   (None, 4, 4, 128)     0           Regularization
-----------------------------------------------------------------------------------------
Flatten                          (None, 2048)          0           Vectorization
Dense (128 units, ReLU)          (None, 128)           262,272     High-level feature combinations
BatchNormalization               (None, 128)           512         Activation stabilization
Dropout (0.50)                   (None, 128)           0           Regularization
Dense (10 units, Softmax)        (None, 10)            1,290       Class probability distribution
=========================================================================================
Total params: ~552,000 (Trainable: ~550,000)
```

---

## 5. Project Directory Structure

```
image-recognition-ml/
├── README.md                               # Project documentation & ML guide
├── requirements.txt                        # Pinned dependencies
├── .gitignore                              # Git exclusion rules
│
├── data/
│   ├── raw/                                # Downloaded raw dataset cache
│   └── processed/                          # Processed dataset partitions
│
├── models/
│   ├── image_classifier.keras              # Saved trained Keras model
│   └── metadata.json                       # Hyperparameters, class maps, metrics
│
├── notebooks/
│   └── 01_cifar10_exploration_and_training.ipynb  # Interactive walkthrough
│
├── outputs/
│   ├── plots/                              # Loss/accuracy graphs, confusion matrix
│   ├── metrics/                            # Classification reports, training history
│   └── predictions/                        # Saved inference outputs
│
├── src/
│   ├── __init__.py                         # Package marker
│   ├── config.py                           # Central configuration & hyperparameters
│   ├── data_loader.py                      # Dataset loading & stratified splitting
│   ├── preprocessing.py                    # Normalization, augmentation, tf.data
│   ├── model.py                            # CNN architecture definition
│   ├── train.py                            # Training pipeline with callbacks
│   ├── evaluate.py                         # Test evaluation & metrics calculation
│   └── predict.py                          # Single-image inference CLI
│
└── tests/
    ├── test_data.py                        # Dataset & normalization tests
    ├── test_model.py                       # CNN architecture & forward pass tests
    └── test_pipeline.py                    # Inference pipeline tests
```

---

## 6. Installation & Environment Setup

### 1. Create Virtual Environment
```bash
# Using Python 3.11 with standard venv:
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Execution Guide

### Step 1: Train the Model
Train the CNN model with automatic dataset loading, normalization, augmentation, and callback monitoring:
```bash
python -m src.train --epochs 30 --batch_size 64 --lr 0.001
```
*Outputs generated:*
- Model checkpoint: `models/image_classifier.keras`
- Training metadata: `models/metadata.json`
- Loss & Accuracy curves: `outputs/plots/training_history.png`
- Epoch metrics log: `outputs/metrics/training_history.json`

### Step 2: Evaluate on the Untouched Test Set
Evaluate the trained model on 10,000 pristine test images to compute accuracy, loss, precision, recall, F1, and the confusion matrix:
```bash
python -m src.evaluate
```
*Outputs generated:*
- Confusion matrix heatmap: `outputs/plots/confusion_matrix.png`
- Detailed classification metrics: `outputs/metrics/classification_report.json`

### Step 3: Run Image Predictions
Classify any image file (PNG/JPEG) and view top-K predictions with confidence percentages:
```bash
python -m src.predict path/to/image.jpg --top_k 3
```
*Example terminal output:*
```
========================================
IMAGE RECOGNITION
========================================

Image: dog_sample.jpg

Prediction:
dog

Confidence:
92.48%

Top 3:
1. dog          92.48%
2. cat           5.12%
3. horse         1.34%

========================================
[NOTE] CIFAR-10 Scope:
This model is trained exclusively on 10 CIFAR-10 classes:
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
It cannot recognize arbitrary objects outside this predefined domain.
========================================
```

### Step 4: Run Automated Unit Tests
Verify dataset loading, normalization bounds, CNN forward passes, and inference pipelines:
```bash
pytest tests/ -v
```

---

## 8. Limitations & Scope

1. **Closed-Set Classification**: The model can classify images **only** into the 10 predefined CIFAR-10 categories. It cannot detect or identify out-of-distribution objects (e.g. guitars, smartphones, humans).
2. **Resolution Constraints**: CIFAR-10 images are $32 \times 32$ pixels. When high-resolution images are supplied, they are downsampled to $32 \times 32$, which can cause fine visual details to be lost.

---

---

## 9. Phase 2: Transfer Learning & Real-World Image Recognition

Phase 2 upgrades the system from small closed-domain CIFAR-10 classification ($32 \times 32$) to high-resolution real-world image recognition ($224 \times 224$) using **Transfer Learning** on ImageNet pre-trained backbones (**MobileNetV2** and **EfficientNetB0**).

```
Phase 1 Baseline (CIFAR-10)               Phase 2 Transfer Learning
• 32x32 pixel input                       • 224x224 high-resolution input
• Custom 3-block CNN (~552K params)       • MobileNetV2 & EfficientNetB0 (~2.6M - 4.4M params)
• 10 toy categories (79.35% Test Acc)     • 2-Stage Training: Frozen Head + Fine-Tuned Top Layers
• Closed domain only                      • Dual-mode CLI: 5-class target domain & 1,000-class ImageNet
```

### 9.1 Why Transfer Learning?

1. **Resolution & Detail Retention**: $32 \times 32$ images lose high-frequency geometric and textural features. $224 \times 224$ images preserve edges, surface textures, and fine morphology.
2. **General Visual Features**: ImageNet pre-trained networks have learned generic visual representations across 1.4 million images (Gabor-like edge filters, texture detectors, shape primitives).
3. **Sample Efficiency**: Training deep vision architectures from scratch on modest datasets causes rapid overfitting. Transfer learning achieves $\ge 90\%$ accuracy with a fraction of the data and compute.

---

### 9.2 Two-Stage Transfer Learning Methodology

```
STAGE A: Feature Extraction
┌─────────────────────────────────────────────────────────────────┐
│ ImageNet Backbone (FROZEN) ──► GAP ──► Dense(256) ──► Softmax(5)│
│ (Weights Locked, LR = 1e-3, 5 Epochs)                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (Restore best weights)
STAGE B: Fine-Tuning
┌─────────────────────────────────────────────────────────────────┐
│ Lower Layers (FROZEN) │ Top Layers (UNFROZEN) ──► Custom Head   │
│ (MobileNetV2: top 54 layers unfrozen; EfficientNetB0: top 58)   │
│ (Very Low LR = 1e-5, 5 Epochs, Early Stopping)                  │
└─────────────────────────────────────────────────────────────────┘
```

1. **Stage A (Feature Extraction)**: The pre-trained backbone base is frozen (`trainable = False`). Only the custom classification head (`GlobalAveragePooling2D` $\to$ `Dense(256, ReLU)` $\to$ `BatchNormalization` $\to$ `Dropout(0.40)` $\to$ `Dense(5, Softmax)`) is trained with Adam ($\text{lr} = 10^{-3}$).
2. **Stage B (Fine-Tuning)**: Lower layers (general edge/texture detectors) remain frozen to prevent catastrophic forgetting, while top convolutional layers are unfrozen and trained at a very low learning rate ($\text{lr} = 10^{-5}$) to specialize high-level semantic representations.

---

### 9.3 Measured Benchmark & Architecture Comparison

All metrics measured on the held-out 551-image test set and local CPU latency benchmark:

| Model Architecture | Input Shape | Top-1 Test Acc | Top-5 Test Acc | Macro F1 | Total Params | Model Size | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetB0 (Selected)** | $224 \times 224 \times 3$ | **90.20%** | **100.00%** | **90.09%** | 4,379,816 | 40.45 MB | ~175.9 ms |
| **MobileNetV2** | $224 \times 224 \times 3$ | **85.12%** | **100.00%** | **85.06%** | 2,588,229 | 27.20 MB | ~180.2 ms |
| **Phase 1 CNN (Baseline)** | $32 \times 32 \times 3$ | 79.35% | 98.70% | 79.07% | 552,010 | 6.74 MB | ~12.4 ms |

---

### 9.4 How to Run Phase 2

#### 1. Train Phase 2 Transfer Learning Models
```bash
python -m src.phase2_train --stage_a_epochs 5 --stage_b_epochs 5
```
Automatically downloads and caches dataset, executes Stage A & Stage B for both MobileNetV2 and EfficientNetB0, saves checkpoints to `models/phase2/`, and exports the best performer to `models/phase2/selected_model.keras`.

#### 2. Run Comprehensive Evaluation & Benchmarks
```bash
python -m src.phase2_evaluate
```
Generates comparison charts (`outputs/phase2/plots/model_comparison.png`), confusion matrix (`outputs/phase2/plots/confusion_matrix.png`), and `outputs/phase2/metrics/comparison_metrics.json`.

#### 3. Run Inference CLI

**Mode 1: Transfer-Learned Target Domain (5 Classes)**
```bash
python -m src.predict_phase2 "path/to/flower_image.jpg" --mode transfer_learned
```
Output:
```
========================================
REAL-WORLD IMAGE RECOGNITION
========================================

Image: 10090824183_d02c613f10_m.jpg
Mode:  TRANSFER_LEARNED

Top 5 predictions:

1. roses                   46.89%
2. dandelion               18.99%
3. daisy                   17.85%
4. tulips                  14.94%
5. sunflowers               1.33%

========================================
[NOTE] Softmax Confidence Notice:
Softmax output represents normalized relative activation across candidate classes,
which can be overconfident for out-of-distribution inputs.
========================================
```

**Mode 2: ImageNet 1,000-Class General Object Recognition**
```bash
python -m src.predict_phase2 "path/to/any_object.jpg" --mode imagenet
```

#### 4. Run Pytest Suite (All 15 Tests)
```bash
pytest tests/ -v
```

---

### 9.5 Out-of-Domain Detection & Softmax Reality

A key ML concept highlighted in Phase 2 is **Softmax Normalization Overconfidence**:
- Because $\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, the probabilities are forced to sum to 1.0 even when the input image belongs to a class completely outside the training distribution (e.g. feeding a picture of a car to a 5-class flower model).
- The Phase 2 CLI implements an **uncertainty warning flag** when maximum confidence is low, and clearly documents the difference between closed-domain transfer learning and open-domain ImageNet-1K recognition.

---

## 10. Phase 3: Advanced Recognition Engine (Classification + Object Detection)

Phase 3 builds a unified, production-oriented inference engine that combines high-resolution **Image Classification** (ImageNet 1,000 classes) with **Multiple Object Detection** (COCO 90 classes with spatial bounding boxes).

```
                            Input Image (JPG, PNG, WEBP, BMP)
                                           │
                                           ▼
                            Image Validation & Normalization
                         (Grayscale->RGB, RGBA Compositing, EXIF)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Image Classification                           Object Detection
        (ImageNet 1,000 Classes)                        (COCO 90 Classes)
        "What is the main subject?"                   "What is where?"
        MobileNetV2 / EfficientNetB0                  SSD MobileNet V2
        Top-1 & Top-5 Probabilities                   Spatial Pixel Bounding Boxes
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                               Unified Recognition Result
                              (Stable JSON Schema + Status)
                                           │
                                           ▼
                              Visual Detection Rendering
                              (outputs/phase3/detections/)
```

---

### 10.1 Key Engine Components

1. **Image Validator ([`src/phase3/image_validator.py`](file:///d:/AHMED%20PROJECTS/ml%20project/src/phase3/image_validator.py))**:
   - Validates file headers, format extensions, and corruption.
   - Converts single-channel grayscale (`L`) and transparent (`RGBA`/`LA`) images safely to 3-channel RGB using white canvas alpha compositing.
   - Automatically applies EXIF orientation normalization.
   - Returns structured `ValidationResult` objects with complete dimensional and aspect-ratio metadata.

2. **Classifier Engine ([`src/phase3/classifier.py`](file:///d:/AHMED%20PROJECTS/ml%20project/src/phase3/classifier.py))**:
   - Exposes ImageNet-1K classification with aspect-preserving center cropping.
   - Provides Top-1 and Top-5 ranked predictions with configurable confidence thresholds.
   - Flags low-confidence predictions as `status: "low_confidence"`.

3. **Object Detector Engine ([`src/phase3/detector.py`](file:///d:/AHMED%20PROJECTS/ml%20project/src/phase3/detector.py))**:
   - Pre-trained on the COCO dataset (90 everyday object categories).
   - Detects multiple objects simultaneously in a single forward pass.
   - Returns exact pixel coordinates `[x1, y1, x2, y2]`, box dimensions `width x height`, and normalized bounds.

4. **Visualizer ([`src/phase3/visualizer.py`](file:///d:/AHMED%20PROJECTS/ml%20project/src/phase3/visualizer.py))**:
   - Renders antialiased bounding boxes and high-contrast category badges onto original-resolution images.
   - Saves rendered detection images to `outputs/phase3/detections/`.

5. **Unified Engine ([`src/phase3/engine.py`](file:///d:/AHMED%20PROJECTS/ml%20project/src/phase3/engine.py))**:
   - High-level interface: `recognize(image_input, mode="all")`.
   - Returns serializable JSON schema consumable by future API/UI layers.

---

### 10.2 Classification vs Object Detection

| Capability | Classification | Object Detection |
| :--- | :--- | :--- |
| **Core Question** | *"What is the overall main subject?"* | *"What objects are present and where?"* |
| **Output Type** | Global class probability distribution | Multiple localized bounding boxes |
| **Model** | MobileNetV2 (ImageNet-1K) | SSD MobileNet V2 (COCO 90) |
| **Supported Classes** | 1,000 fine-grained categories | 90 common object categories |
| **Output Format** | Label + Confidence (%) | Label + Confidence (%) + Bounding Box `[x1, y1, x2, y2]` |

---

### 10.3 Phase 3 CLI Usage

#### Unified Mode (Classification + Detection + Visualizer)
```bash
python -m src.phase3.cli "path/to/image.jpg" --mode all
```

**Example Output:**
```
============================================================
        ADVANCED IMAGE RECOGNITION REPORT (ALL)
============================================================
Image File:   10090824183_d02c613f10_m.jpg (179x240 RGB)
Total Latency: 1822.16 ms
------------------------------------------------------------

1. IMAGE CLASSIFICATION (ImageNet 1,000 Classes):
   Model:        MobileNetV2 (308.89 ms)
   Primary:      daisy (30.17%) [LOW_CONFIDENCE]

   Top 5 Ranked Predictions:
     1. daisy                      30.17%
     2. earthstar                  11.06%
     3. hair slide                  8.70%
     4. bee                         7.35%
     5. ant                         2.70%

   [!] UNCERTAINTY NOTICE:
       Confidence (30.17%) is below 40% threshold.

------------------------------------------------------------
2. OBJECT DETECTION (COCO 90 Object Classes):
   Model:        SSD-MobileNetV2-COCO (1402.64 ms)
   Detected:     2 object(s)

   [1] POTTED PLANT (58.4%) -> Box: [x1=7, y1=16, x2=171, y2=239] (164x223 px)
   [2] VASE (40.1%) -> Box: [x1=14, y1=89, x2=143, y2=237] (129x148 px)

   Rendered Visualization: outputs/phase3/detections/10090824183_d02c613f10_m_detected.png
============================================================
```

#### JSON Output Mode
```bash
python -m src.phase3.cli "path/to/image.jpg" --mode all --json
```

---

### 10.4 Performance Benchmarks

Measured on host CPU across standard resolutions (15 iterations per resolution):

| Resolution | Dimensions | Classifier Latency | Detector Latency | Unified Pipeline (with Rendering) |
| :--- | :---: | :---: | :---: | :---: |
| **VGA** | $640 \times 480$ | 158.3 ms (6.3 FPS) | 971.6 ms | 1263.6 ms (0.8 FPS) |
| **HD** | $1280 \times 720$ | 135.2 ms (7.4 FPS) | 1043.0 ms | 1476.8 ms (0.7 FPS) |
| **Square** | $800 \times 800$ | 139.0 ms (7.2 FPS) | 1031.7 ms | 1606.3 ms (0.6 FPS) |
| **Portrait** | $600 \times 900$ | 228.6 ms (4.4 FPS) | 1164.7 ms | 1569.3 ms (0.6 FPS) |
| **Small** | $300 \times 300$ | 221.6 ms (4.5 FPS) | 1433.4 ms | 1679.9 ms (0.6 FPS) |

---

### 10.5 Run Automated Tests

Verify Phase 1, Phase 2, and Phase 3 suites with full real-model execution:
```bash
pytest tests/ -v
```

---

## 11. Phase 4 — Production FastAPI Serving Layer

Phase 4 exposes the complete ML Image Recognition & Object Detection Engine over a production-oriented HTTP REST API using **FastAPI**, **Uvicorn**, and **Pydantic V2**.

### 11.1 Architecture & Request Flow

```text
Client (Web / Mobile / cURL)
             ↓
 FastAPI HTTP Application (/api/v1)
             ↓
 Request Validation & Streaming Upload Limit (10MB)
             ↓
 Phase 3 Image Normalization & Validation (RGB / EXIF / Alpha compositing)
             ↓
 Threadpool Dispatch (run_in_threadpool) for Non-Blocking CPU ML Inference
             ↓
 MobileNetV2 (ImageNet-1K) + SSD-MobileNetV2 (COCO-80)
             ↓
 Normalized JSON-Serializable Pydantic Response
```

### 11.2 Environment Variables & Configuration

Configure via environment or `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_HOST` | `127.0.0.1` | Host interface to bind server |
| `API_PORT` | `8000` | Port to bind server |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum allowed multipart image upload size |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for web clients |
| `LOG_LEVEL` | `INFO` | API logging level |

### 11.3 Starting the Server

```bash
# Start FastAPI application with Uvicorn
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive Documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

### 11.4 API Endpoints Summary

| Method | Endpoint | Description | Request Type | Response Model |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service liveness probe | None | `HealthResponse` |
| `GET` | `/api/v1/ready` | ML engine readiness probe | None | `ReadyResponse` |
| `GET` | `/api/v1/models` | Model metadata & specs | None | `ModelInfoResponse` |
| `POST` | `/api/v1/recognize` | Unified classification & detection | `multipart/form-data` | `RecognitionResponse` |
| `POST` | `/api/v1/classify` | Standalone ImageNet classification | `multipart/form-data` | `ClassificationResult` |
| `POST` | `/api/v1/detect` | Standalone COCO object detection | `multipart/form-data` | `DetectionResult` |

---

### 11.5 Example Requests & Responses

#### 1. Unified Recognition (`POST /api/v1/recognize`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recognize" \
  -F "image=@sample.jpg" \
  -F "mode=all" \
  -F "classification_threshold=0.3" \
  -F "detection_threshold=0.3"
```
**Response (200 OK)**:
```json
{
  "success": true,
  "mode": "all",
  "image": {
    "file_name": "sample.jpg",
    "file_path": "",
    "width": 1280,
    "height": 720,
    "channels": 3,
    "format": "JPEG",
    "aspect_ratio": 1.778,
    "file_size_bytes": 154200
  },
  "metadata": {
    "engine": "AdvancedRecognitionEngine-v3",
    "classifier_model": "MobileNetV2",
    "detector_model": "SSD-MobileNetV2-COCO",
    "classification_threshold": 0.3,
    "detection_threshold": 0.3,
    "total_inference_ms": 942.5
  },
  "classification": {
    "success": true,
    "mode": "classification",
    "status": "confident",
    "top1": {
      "label": "golden retriever",
      "confidence": 0.8842,
      "confidence_percent": 88.42
    },
    "top5": [
      {
        "label": "golden retriever",
        "confidence": 0.8842,
        "confidence_percent": 88.42
      }
    ],
    "model": "MobileNetV2",
    "inference_ms": 115.2
  },
  "detection": {
    "success": true,
    "mode": "detection",
    "status": "objects_detected",
    "count": 1,
    "objects": [
      {
        "label": "dog",
        "class_id": 18,
        "confidence": 0.9234,
        "confidence_percent": 92.34,
        "box": {
          "x1": 256,
          "y1": 108,
          "x2": 960,
          "y2": 612,
          "width": 704,
          "height": 504,
          "normalized": [0.15, 0.20, 0.85, 0.75]
        }
      }
    ],
    "model": "SSD-MobileNetV2-COCO",
    "inference_ms": 782.1
  }
}
```

#### 2. Standalone Classification (`POST /api/v1/classify`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify" \
  -F "image=@sample.jpg" \
  -F "top_k=5"
```

#### 3. Standalone Detection (`POST /api/v1/detect`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
  -F "image=@sample.jpg" \
  -F "threshold=0.4"
```

---

### 11.6 Normalized Error Response Format

All error conditions return uniform JSON structures with descriptive codes and HTTP status codes:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The uploaded file is not a valid image.",
    "request_id": "req-7b89f2a012"
  }
}
```

| HTTP Status | Error Code | Trigger Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `EMPTY_FILE` | 0-byte file uploaded |
| `400 Bad Request` | `CORRUPTED_IMAGE` | Unidentified or malformed image binary |
| `400 Bad Request` | `UNSUPPORTED_FORMAT` | Extension not in `.jpg, .jpeg, .png, .webp, .bmp` |
| `413 Content Too Large` | `FILE_TOO_LARGE` | Upload exceeds `MAX_UPLOAD_SIZE_MB` (10MB) |
| `422 Unprocessable` | `INVALID_MODE` | Mode not in `all, classification, detection` |
| `422 Unprocessable` | `INVALID_REQUEST_PARAMETERS` | Missing image field or invalid form params |
| `503 Service Unavailable`| `MODEL_UNAVAILABLE` | ML models failed to load at startup |

---

### 11.7 Model Lifecycle & Memory Management

- **Single Initialization**: Loaded during the FastAPI application lifespan into `app.state.engine`.
- **Zero Per-Request Reloading**: Models stay active in host memory across requests.
- **Threadpool Offloading**: CPU-bound TensorFlow predictions are dispatched via `starlette.concurrency.run_in_threadpool`, ensuring the async event loop remains fully responsive under concurrent traffic.

---

### 11.8 Performance & Latency Breakdown

| Operation | Direct CLI (ms) | HTTP Serving API (ms) | API Overhead (ms) |
| :--- | :---: | :---: | :---: |
| **ImageNet Classification** | 114.7 ms | 190.4 ms | +75.7 ms |
| **COCO Object Detection** | 799.5 ms | 1124.2 ms | +324.6 ms |
| **Unified All-Mode** | 914.8 ms | 1147.9 ms | +233.1 ms |

---

### 11.9 Complete Regression Test Suite (48 Tests)

```bash
pytest tests/ -v
```
```text
============================= test session starts =============================
collected 48 items

tests/test_api_health.py::test_health_endpoint PASSED                    [  2%]
tests/test_api_health.py::test_readiness_endpoint PASSED                 [  4%]
tests/test_api_health.py::test_models_metadata_endpoint PASSED           [  6%]
tests/test_api_health.py::test_openapi_schema_endpoint PASSED            [  8%]
tests/test_api_health.py::test_docs_redirect PASSED                      [ 10%]
tests/test_api_recognition.py::test_recognize_all_mode PASSED            [ 12%]
tests/test_api_recognition.py::test_recognize_classification_mode PASSED [ 14%]
tests/test_api_recognition.py::test_recognize_detection_mode PASSED      [ 16%]
tests/test_api_recognition.py::test_standalone_classify_endpoint PASSED  [ 18%]
tests/test_api_recognition.py::test_standalone_detect_endpoint PASSED    [ 20%]
tests/test_api_recognition.py::test_concurrent_api_requests PASSED       [ 22%]
tests/test_api_validation.py::test_missing_image_file_returns_422 PASSED [ 25%]
tests/test_api_validation.py::test_empty_zero_byte_file_returns_400 PASSED [ 27%]
tests/test_api_validation.py::test_corrupted_image_file_returns_400 PASSED [ 29%]
tests/test_api_validation.py::test_unsupported_file_extension_returns_400 PASSED [ 31%]
tests/test_api_validation.py::test_invalid_mode_returns_422 PASSED       [ 33%]
tests/test_api_validation.py::test_oversized_upload_returns_413 PASSED   [ 35%]
tests/test_api_validation.py::test_custom_request_id_header PASSED       [ 37%]
tests/test_data.py::test_class_names_configuration PASSED                [ 39%]
tests/test_data.py::test_normalize_images_range_and_dtype PASSED         [ 41%]
tests/test_data.py::test_normalize_idempotence PASSED                    [ 43%]
tests/test_model.py::test_model_build_and_shapes PASSED                  [ 45%]
tests/test_model.py::test_model_forward_pass PASSED                      [ 47%]
tests/test_phase2_data.py::test_phase2_class_names_and_count PASSED      [ 50%]
tests/test_phase2_data.py::test_create_phase2_splits_ratios PASSED       [ 52%]
tests/test_phase2_data.py::test_preprocess_phase2_image_shape_and_aspect_ratio PASSED [ 54%]
tests/test_phase2_models.py::test_mobilenetv2_model_build_and_forward_pass PASSED [ 56%]
tests/test_phase2_models.py::test_mobilenetv2_unfreezing PASSED          [ 58%]
tests/test_phase2_pipeline.py::test_compute_top_k_accuracy PASSED        [ 60%]
tests/test_phase2_pipeline.py::test_format_phase2_output_uncertainty PASSED [ 62%]
tests/test_phase3_engine.py::test_classifier_output_schema_and_top_k PASSED [ 64%]
tests/test_phase3_engine.py::test_classifier_confidence_threshold_behavior PASSED [ 66%]
tests/test_phase3_engine.py::test_visualization_generation PASSED        [ 68%]
tests/test_phase3_engine.py::test_unified_engine_classification_mode PASSED [ 70%]
tests/test_phase3_engine.py::test_unified_engine_detection_mode PASSED   [ 72%]
tests/test_phase3_engine.py::test_unified_engine_all_mode_and_bounding_boxes PASSED [ 75%]
tests/test_phase3_engine.py::test_unified_engine_invalid_input_handling PASSED [ 77%]
tests/test_phase3_validation.py::test_valid_jpg_validation PASSED        [ 79%]
tests/test_phase3_validation.py::test_valid_png_validation PASSED        [ 81%]
tests/test_phase3_validation.py::test_grayscale_to_rgb_conversion PASSED [ 83%]
tests/test_phase3_validation.py::test_rgba_to_rgb_compositing PASSED     [ 85%]
tests/test_phase3_validation.py::test_missing_file_handling PASSED       [ 87%]
tests/test_phase3_validation.py::test_empty_zero_byte_file PASSED        [ 89%]
tests/test_phase3_validation.py::test_unsupported_extension_handling PASSED [ 91%]
tests/test_phase3_validation.py::test_numpy_array_input_validation PASSED [ 93%]
tests/test_pipeline.py::test_preprocess_single_image PASSED              [ 95%]
tests/test_pipeline.py::test_preprocess_grayscale_conversion PASSED      [ 97%]
tests/test_pipeline.py::test_preprocess_nonexistent_file PASSED          [100%]

============================= 48 passed in 102.98s =============================
```

---

## 12. Phase 5 — Interactive Web UI (Next.js + TypeScript + Tailwind)

Phase 5 introduces a responsive, interactive web application built with **Next.js 14**, **TypeScript**, **Tailwind CSS**, and **Lucide Icons** that directly interfaces with the FastAPI serving layer.

### 12.1 Web UI Architecture

```text
Browser (Next.js Client / React 18)
               ↓
 Centralized API Client (src/lib/api.ts)
               ↓
 FastAPI HTTP Serving Layer (http://127.0.0.1:8000/api/v1)
               ↓
 Phase 3 Unified Engine (MobileNetV2 + SSD-MobileNetV2)
               ↓
 Interactive Detection Canvas & Classification Dashboard
```

### 12.2 Features & Capabilities

- **Drag & Drop Upload Zone**: Client-side format filtering (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) and 10MB file size limit validation.
- **Image Preview & Pre-Analysis Meta**: Displays image thumbnail, file size, dimensions, and aspect ratio before submitting inference.
- **Multi-Mode Selector**:
  - `All`: Unified ImageNet-1K classification + COCO-80 object detection.
  - `Classification`: Top-1 primary prediction + Top-5 ranked candidate list.
  - `Object Detection`: Entity bounding boxes with spatial coordinate overlays.
- **Interactive Bounding Box Viewer**:
  - Pixel-perfect, responsive SVG/CSS bounding boxes overlaying the rendered image.
  - Bidirectional hover synchronization: Hovering over a bounding box highlights the corresponding object card, and hovering over an object card highlights the bounding box on the image canvas.
- **Classification Status & Uncertainty**: Displays primary match with confidence percentage and a dedicated notice for low-confidence classifications.
- **Technical Telemetry Drawer**: Collapsible telemetry showing total API latency, internal model execution ms, image normalization specs, and model architectures.
- **Live API Status Pill**: Real-time health indicator (`Connected`, `Unavailable`, `Checking...`) with manual refresh button.

### 12.3 Starting the Full Application

#### 1. Start the FastAPI Backend
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

#### 2. Start the Next.js Frontend
```bash
cd frontend
npm install
npm run dev -- -p 3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 12.4 Frontend Verification & Testing

```bash
cd frontend
npm test
npm run build
```
- **Build Output**: Next.js production build compiled with 0 TypeScript/ESLint errors.
- **Unit Tests**: 8/8 tests passing covering API client error mapping, bounding box scaling math, and open-vocabulary tag constraints.
- **Full Backend Regression**: 60/60 Python tests passing.

---

## 13. Phase 6A — Zero-Shot Open-Vocabulary Recognition (OpenCLIP ViT-B/32)

Phase 6A upgrades the image recognition system from fixed-vocabulary classifiers (CIFAR-10, ImageNet-1K, COCO-80) to **arbitrary open-vocabulary zero-shot recognition** using **OpenCLIP ViT-B/32** (`laion2b_s34b_b79k`).

### 13.1 Architecture & Design Decisions

```text
Uploaded Image (RGB, Any Aspect Ratio)
         │
         ▼
OpenCLIP Vision Transformer (ViT-B/32) ──► Image Embedding e_img ∈ R^512 (L2-Normalized)
                                                                 │
Candidate Text Queries ["rose", "laptop", ...]                   ├──► Cosine Similarity (e_img · e_txt)
         │                                                       │
         ▼                                                       ▼
OpenCLIP Text Transformer ───────────────► Text Embeddings e_txt ∈ R^512  Ranked Semantic Scores [-1.0, 1.0]
```

- **Backbone**: `ViT-B/32` pre-trained on LAION-2B (`laion2b_s34b_b79k`).
- **Hardware Architecture**: CPU-first with `torch.set_num_threads(4)` explicitly matching the 4 physical CPU cores of the host Intel Core i5 processor.
- **Strict Separation of Metrics**: OpenCLIP produces dot products of normalized unit embeddings ($\mathbf{e}_I \cdot \mathbf{e}_T$). Raw scores are presented as **Cosine Similarity** meters—distinct from closed-domain softmax probability percentages.
- **Prompt Engineering**: Dynamic prompt templating (`"a photo of a {}"`) applied automatically to boost semantic separation.
- **Additive Preservation**: Phases 1–5 models, endpoints (`/classify`, `/detect`, `/recognize`), and visualizers remain 100% untouched.

### 13.2 API Endpoint: `POST /api/v1/open-vocabulary`

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | `UploadFile` | Yes | Multipart image file (JPG, PNG, WEBP, BMP, $\le 10\text{ MB}$) |
| `text_queries` | `string` | Yes | Comma-separated or JSON list of candidate concepts (1–20 concepts) |
| `top_k` | `int` | No | Number of top ranked candidates to return (default: 5, max: 20) |
| `similarity_threshold`| `float` | No | Optional minimum cosine similarity filter ($\in [-1.0, 1.0]$) |
| `prompt_template` | `string` | No | Prompt template string (default: `"a photo of a {}"`) |

#### Example cURL Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/open-vocabulary" \
  -F "image=@sample_rose.jpg" \
  -F "text_queries=rose, sunflower, sports car, laptop, airplane" \
  -F "top_k=5"
```

#### Example Response:
```json
{
  "success": true,
  "mode": "open_vocabulary",
  "image": {
    "file_name": "sample_rose.jpg",
    "width": 500,
    "height": 333,
    "channels": 3,
    "format": "JPEG",
    "aspect_ratio": 1.502,
    "file_size_bytes": 105421
  },
  "open_vocabulary": {
    "model": "OpenCLIP-ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "prompt_template": "a photo of a {}",
    "queries": ["rose", "sunflower", "sports car", "laptop", "airplane"],
    "results": [
      { "query": "rose", "similarity_score": 0.2754, "rank": 1 },
      { "query": "sunflower", "similarity_score": 0.1504, "rank": 2 },
      { "query": "airplane", "similarity_score": 0.1060, "rank": 3 },
      { "query": "laptop", "similarity_score": 0.0904, "rank": 4 },
      { "query": "sports car", "similarity_score": 0.0753, "rank": 5 }
    ],
    "top_match": { "query": "rose", "similarity_score": 0.2754, "rank": 1 },
    "no_match": false,
    "min_similarity_threshold": 0.20,
    "inference_ms": 381.5,
    "preprocessing_ms": 11.2,
    "total_ms": 392.7
  },
  "metadata": {
    "engine": "AdvancedRecognitionEngine-v3-OpenVocab",
    "open_vocab_model": "OpenCLIP-ViT-B-32",
    "total_inference_ms": 392.7
  }
}
```

### 13.3 CPU Latency Benchmarks

Measured on Intel Core i5-8265U (4 physical CPU cores, 8GB RAM):

| Query Configuration | Average CPU Latency | Min Latency | Max Latency |
| :--- | :--- | :--- | :--- |
| **1 Query** | **217.3 ms** | 191.1 ms | 232.3 ms |
| **5 Queries** | **392.8 ms** | 381.8 ms | 407.1 ms |
| **10 Queries** | **696.1 ms** | 669.2 ms | 723.3 ms |
| **20 Queries** | **1,278.7 ms** | 1,230.0 ms | 1,373.5 ms |

### 13.4 Negative Controls & Out-of-Domain Separation

When evaluating a ground-truth image of a **Rose** against purely distractor categories (`refrigerator`, `submarine`, `space shuttle`, `microwave`, `bulldozer`):
- **True Class Score (`rose`)**: `+0.2754`
- **Max Distractor Score (`microwave`)**: `+0.1163`
- **Mean Distractor Score**: `+0.0975`
- **Semantic Discrimination Margin**: **`+0.1591`** ($+136.8\%$ separation over distractor ceiling)

### 13.5 Complete Test Suite Verification

```text
================== 72 passed, 1 warning in 459.66s (0:07:39) ==================
```
- **Phase 1 Baseline CNN Tests**: 4/4 passing
- **Phase 2 Transfer Learning & Pipeline Tests**: 7/7 passing
- **Phase 3 Unified Engine & Visualizer Tests**: 15/15 passing
- **Phase 4 FastAPI Health, Ready, Models, Validation Tests**: 22/22 passing
- **Phase 6A OpenCLIP Classifier & OpenVocab API Tests**: 12/12 passing
- **Phase 6B OWL-ViT Open-Vocabulary Object Detection Tests**: 12/12 passing
- **Frontend Unit Tests (`node --test`)**: 10/10 passing
- **Frontend Build (`npm run build`)**: Compiled successfully with 0 errors

---

## 14. Phase 6B — Production Open-Vocabulary Object Detection (OWL-ViT)

Phase 6B advances the vision serving system from image-level zero-shot recognition to **open-vocabulary object localization and bounding box prediction** using Google's **OWL-ViT base patch32** (`google/owlvit-base-patch32`).

### 14.1 Architecture & Lazy-Load Model Lifecycle

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
                      |  - Memory Release & Unload  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  OWLViTDetector (CPU-4T)    |
                      |  - google/owlvit-base-patch32|
                      |  - Cross-Attention Backbone |
                      |  - Bounding Box Regression  |
                      +--------------+--------------+
```

### 14.2 Technical Specifications

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Model ID** | `google/owlvit-base-patch32` | Vision Transformer ViT-B/32 backbone + cross-attention multi-modal projection heads |
| **Parameters** | **153,231,879** (~153.2M) | Pre-trained for open-vocabulary grounded object detection |
| **Disk Footprint** | **586.10 MB** | Stored in Hugging Face cache |
| **Process Memory** | **~1,012 MB** Working Set | **Lazy-loaded on first request** (not resident on startup) |
| **Execution** | **CPU Only** | Pinned to 4 CPU worker threads (`torch.set_num_threads(4)`) |
| **Single-Query Latency** | **~1,281 ms** | Single forward vision pass ($O(1)$) with cross-attention query heads |
| **20-Query Latency** | **~1,499 ms** | Flat scaling (+17% time for 20x queries) |

### 14.3 Dedicated Endpoints

#### `POST /api/v1/open-vocabulary/detect`
- **Payload (`multipart/form-data`)**:
  - `image`: Multipart file upload (JPG, PNG, WEBP, BMP, $\le 10\text{MB}$)
  - `text_queries`: Comma-separated or JSON list of candidate object concepts (max 20)
  - `top_k`: Maximum bounding boxes to return (1–50, default 20)
  - `score_threshold`: Minimum alignment confidence cutoff [0.0, 1.0] (default 0.10)
  - `prompt_template`: Prompt template string containing `{}` (default `"a photo of a {}"`)

- **Example Response**:
```json
{
  "success": true,
  "mode": "open_vocabulary_detection",
  "image": {
    "file_name": "street_scene.jpg",
    "width": 1080,
    "height": 720,
    "format": "JPEG",
    "file_size_bytes": 145200
  },
  "model": {
    "name": "OWL-ViT",
    "model_id": "google/owlvit-base-patch32",
    "parameters": 153231879,
    "device": "cpu"
  },
  "summary": {
    "total_queries": 3,
    "total_detections": 2,
    "score_threshold": 0.10,
    "prompt_template": "a photo of a {}"
  },
  "queries": [
    {
      "query": "bus",
      "prompt_applied": "a photo of a bus",
      "count": 1,
      "top_score": 0.6429,
      "detections": [...]
    }
  ],
  "detections": [
    {
      "detection_id": "det-1",
      "query": "bus",
      "prompt_used": "a photo of a bus",
      "score": 0.6429,
      "box": {
        "x_min": 12,
        "y_min": 180,
        "x_max": 840,
        "y_max": 650,
        "width": 828,
        "height": 470
      },
      "box_normalized": {
        "x_min": 0.0111,
        "y_min": 0.2500,
        "x_max": 0.7778,
        "y_max": 0.9028
      }
    }
  ],
  "timing": {
    "preprocessing_ms": 14.2,
    "inference_ms": 1280.5,
    "total_ms": 1294.7
  }
}
```

#### `POST /api/v1/open-vocabulary/unload`
Explicitly releases OWL-ViT weights from RAM and triggers `gc.collect()` for memory management.

#### `GET /api/v1/open-vocabulary/status`
Returns live runtime telemetry (whether model is loaded, load count, total inferences, device).




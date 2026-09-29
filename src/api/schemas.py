"""Pydantic Request and Response Schemas matching Phase 3 ML Engine output."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Liveness check response."""
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["image-recognition-api"])
    version: str = Field(..., examples=["1.0.0"])
    engine: str = Field(..., examples=["ready"])
    models: Dict[str, bool] = Field(
        ...,
        examples=[{"classifier": True, "detector": True}],
    )


class ReadyResponse(BaseModel):
    """Readiness probe response indicating model initialization status."""
    status: str = Field(..., examples=["ready"])
    engine: str = Field(..., examples=["AdvancedRecognitionEngine-v3"])
    models_ready: bool = Field(..., examples=[True])
    classifier_loaded: bool = Field(..., examples=[True])
    detector_loaded: bool = Field(..., examples=[True])


# ---------------------------------------------------------------------------
# Model Metadata Schemas
# ---------------------------------------------------------------------------

class ClassifierModelInfo(BaseModel):
    """Metadata for classification backbone."""
    model: str = Field(..., examples=["MobileNetV2"])
    vocabulary: str = Field(..., examples=["ImageNet-1K"])
    classes: int = Field(..., examples=[1000])
    input_shape: List[int] = Field(..., examples=[[224, 224, 3]])


class DetectorModelInfo(BaseModel):
    """Metadata for object detection backbone."""
    model: str = Field(..., examples=["SSD-MobileNetV2-COCO"])
    vocabulary: str = Field(..., examples=["COCO-2017"])
    semantic_categories: int = Field(..., examples=[80])
    input_format: str = Field(..., examples=["uint8 RGB [0, 255]"])


class OpenVocabModelInfo(BaseModel):
    """Metadata for Open-Vocabulary vision-language backbone."""
    model: str = Field(..., examples=["OpenCLIP-ViT-B-32"])
    pretrained: str = Field(..., examples=["laion2b_s34b_b79k"])
    supports_text_queries: bool = Field(default=True, examples=[True])
    supports_bounding_boxes: bool = Field(default=False, examples=[False])
    device: str = Field(default="cpu", examples=["cpu"])
    max_queries: int = Field(default=20, examples=[20])
    max_query_length: int = Field(default=128, examples=[128])


class OpenVocabDetectionModelInfo(BaseModel):
    """Metadata for Open-Vocabulary Object Detection backbone (OWL-ViT)."""
    model: str = Field(..., examples=["OWL-ViT base patch32"])
    model_id: str = Field(..., examples=["google/owlvit-base-patch32"])
    supports_text_queries: bool = Field(default=True, examples=[True])
    supports_bounding_boxes: bool = Field(default=True, examples=[True])
    loaded: bool = Field(default=False, examples=[False])
    device: str = Field(default="cpu", examples=["cpu"])
    lazy_loaded: bool = Field(default=True, examples=[True])
    max_queries: int = Field(default=20, examples=[20])
    max_query_length: int = Field(default=128, examples=[128])


class RegionIntelligenceModelInfo(BaseModel):
    """Metadata for Grounded Region-Level Open-Vocabulary Recognition Cascade."""
    available: bool = Field(default=True, examples=[True])
    detector: str = Field(default="OWL-ViT base patch32", examples=["OWL-ViT base patch32"])
    recognizer: str = Field(default="OpenCLIP-ViT-B-32", examples=["OpenCLIP-ViT-B-32"])
    lazy_loaded: bool = Field(default=True, examples=[True])
    max_regions: int = Field(default=20, examples=[20])
    crop_padding_percent: int = Field(default=10, examples=[10])


class ModelInfoResponse(BaseModel):
    """Response containing system model architectures and supported formats."""
    classifier: ClassifierModelInfo
    detector: DetectorModelInfo
    open_vocabulary: Optional[OpenVocabModelInfo] = None
    open_vocabulary_detection: Optional[OpenVocabDetectionModelInfo] = None
    region_intelligence: Optional[RegionIntelligenceModelInfo] = None
    device_info: Optional[Dict[str, Any]] = None
    supported_formats: List[str] = Field(..., examples=[["jpg", "jpeg", "png", "webp", "bmp"]])
    supported_modes: List[str] = Field(..., examples=[["all", "classification", "detection", "open_vocabulary", "open_vocabulary_detection", "region_recognition"]])
    max_upload_size_mb: int = Field(..., examples=[10])


# ---------------------------------------------------------------------------
# Recognition & Inference Schemas (Direct mapping from Phase 3 Engine)
# ---------------------------------------------------------------------------

class ImageInfo(BaseModel):
    """Validated image metadata from Phase 3 validator."""
    file_name: str = Field(..., examples=["sample.jpg"])
    file_path: Optional[str] = Field(default="", examples=[""])
    width: int = Field(..., examples=[1280])
    height: int = Field(..., examples=[720])
    channels: int = Field(..., examples=[3])
    format: str = Field(..., examples=["JPEG"])
    aspect_ratio: float = Field(..., examples=[1.778])
    file_size_bytes: int = Field(..., examples=[154200])


class PredictionCandidate(BaseModel):
    """Candidate label and confidence score."""
    label: str = Field(..., examples=["golden retriever"])
    confidence: float = Field(..., examples=[0.8842])
    confidence_percent: float = Field(..., examples=[88.42])


class ClassificationResult(BaseModel):
    """Classification inference output matching Phase 3 schema."""
    success: bool = Field(..., examples=[True])
    mode: str = Field(default="classification", examples=["classification"])
    status: str = Field(..., examples=["confident"])
    top1: PredictionCandidate
    top5: List[PredictionCandidate] = Field(default_factory=list)
    predictions: List[PredictionCandidate] = Field(default_factory=list)
    confidence_threshold: float = Field(..., examples=[0.40])
    is_confident: bool = Field(..., examples=[True])
    inference_ms: float = Field(..., examples=[42.5])
    preprocessing_ms: float = Field(..., examples=[3.2])
    total_ms: float = Field(..., examples=[45.7])
    model: str = Field(..., examples=["MobileNetV2"])
    input_dimensions: str = Field(..., examples=["224x224x3"])
    error: Optional[str] = None
    error_code: Optional[str] = None


class BoxCoordinates(BaseModel):
    """Bounding box pixel coordinates and normalized bounds."""
    x1: int = Field(..., examples=[256])
    y1: int = Field(..., examples=[108])
    x2: int = Field(..., examples=[960])
    y2: int = Field(..., examples=[612])
    width: int = Field(..., examples=[704])
    height: int = Field(..., examples=[504])
    normalized: List[float] = Field(..., examples=[[0.15, 0.20, 0.85, 0.75]])


class DetectedObject(BaseModel):
    """Detected entity with label, class ID, and bounding box."""
    label: str = Field(..., examples=["dog"])
    class_id: int = Field(..., examples=[18])
    confidence: float = Field(..., examples=[0.9234])
    confidence_percent: float = Field(..., examples=[92.34])
    box: BoxCoordinates


class Dimensions(BaseModel):
    """Original image dimensions."""
    width: int = Field(..., examples=[1280])
    height: int = Field(..., examples=[720])


class DetectionResult(BaseModel):
    """Object detection inference output matching Phase 3 schema."""
    success: bool = Field(..., examples=[True])
    mode: str = Field(default="detection", examples=["detection"])
    status: str = Field(..., examples=["objects_detected"])
    count: int = Field(..., examples=[1])
    objects: List[DetectedObject] = Field(default_factory=list)
    confidence_threshold: float = Field(..., examples=[0.40])
    rendered_image_path: Optional[str] = None
    inference_ms: float = Field(..., examples=[68.2])
    preprocessing_ms: float = Field(..., examples=[5.1])
    total_ms: float = Field(..., examples=[73.3])
    model: str = Field(..., examples=["SSD-MobileNetV2-COCO"])
    image_dimensions: Optional[Dimensions] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class RecognitionMetadata(BaseModel):
    """Operational telemetry and model metadata for unified inference."""
    engine: str = Field(..., examples=["AdvancedRecognitionEngine-v3"])
    classifier_model: Optional[str] = Field(default=None, examples=["MobileNetV2"])
    detector_model: Optional[str] = Field(default=None, examples=["SSD-MobileNetV2-COCO"])
    classification_threshold: float = Field(..., examples=[0.40])
    detection_threshold: float = Field(..., examples=[0.40])
    total_inference_ms: float = Field(..., examples=[112.7])


class RecognitionResponse(BaseModel):
    """Unified response contract for /api/v1/recognize."""
    success: bool = Field(..., examples=[True])
    mode: str = Field(..., examples=["all"])
    image: ImageInfo
    metadata: RecognitionMetadata
    classification: Optional[ClassificationResult] = None
    detection: Optional[DetectionResult] = None


# ---------------------------------------------------------------------------
# Open-Vocabulary Schemas (Phase 6A)
# ---------------------------------------------------------------------------

class QueryRankItem(BaseModel):
    """Individual candidate query with its raw cosine similarity score and rank."""
    query: str = Field(..., examples=["rose"])
    similarity_score: float = Field(..., examples=[0.2754])
    rank: int = Field(..., examples=[1])


class OpenVocabResult(BaseModel):
    """OpenCLIP zero-shot inference output."""
    model: str = Field(..., examples=["OpenCLIP-ViT-B-32"])
    pretrained: str = Field(..., examples=["laion2b_s34b_b79k"])
    prompt_template: str = Field(..., examples=["a photo of a {}"])
    queries: List[str] = Field(..., examples=[["rose", "sunflower", "automobile"]])
    results: List[QueryRankItem] = Field(default_factory=list)
    top_match: Optional[QueryRankItem] = None
    no_match: bool = Field(default=False, examples=[False])
    min_similarity_threshold: float = Field(..., examples=[0.20])
    inference_ms: float = Field(..., examples=[320.5])
    preprocessing_ms: float = Field(..., examples=[15.2])
    total_ms: float = Field(..., examples=[335.7])


class OpenVocabMetadata(BaseModel):
    """Telemetry for open-vocabulary request."""
    engine: str = Field(..., examples=["AdvancedRecognitionEngine-v3-OpenVocab"])
    open_vocab_model: str = Field(..., examples=["OpenCLIP-ViT-B-32"])
    total_inference_ms: float = Field(..., examples=[350.2])


class OpenVocabResponse(BaseModel):
    """Response contract for /api/v1/open-vocabulary."""
    success: bool = Field(default=True, examples=[True])
    mode: str = Field(default="open_vocabulary", examples=["open_vocabulary"])
    image: ImageInfo
    open_vocabulary: OpenVocabResult
    metadata: OpenVocabMetadata


# ---------------------------------------------------------------------------
# Open-Vocabulary Object Detection Schemas (Phase 6B - OWL-ViT)
# ---------------------------------------------------------------------------

class OWLViTPixelBox(BaseModel):
    """Absolute pixel coordinate bounding box."""
    x_min: int = Field(..., examples=[120])
    y_min: int = Field(..., examples=[85])
    x_max: int = Field(..., examples=[490])
    y_max: int = Field(..., examples=[512])
    width: int = Field(..., examples=[370])
    height: int = Field(..., examples=[427])


class OWLViTNormalizedBox(BaseModel):
    """Normalized [0.0, 1.0] bounding box coordinates."""
    x_min: float = Field(..., examples=[0.096])
    y_min: float = Field(..., examples=[0.118])
    x_max: float = Field(..., examples=[0.383])
    y_max: float = Field(..., examples=[0.711])


class OWLViTDetectionItem(BaseModel):
    """Individual object detection instance from OWL-ViT."""
    detection_id: str = Field(..., examples=["det-1"])
    query: str = Field(..., examples=["dog"])
    prompt_used: str = Field(..., examples=["a photo of a dog"])
    score: float = Field(..., examples=[0.7184])
    box: OWLViTPixelBox
    box_normalized: OWLViTNormalizedBox


class OWLViTQueryGroup(BaseModel):
    """Detections grouped by candidate text query."""
    query: str = Field(..., examples=["dog"])
    prompt_applied: str = Field(..., examples=["a photo of a dog"])
    count: int = Field(..., examples=[1])
    top_score: float = Field(..., examples=[0.7184])
    detections: List[OWLViTDetectionItem] = Field(default_factory=list)


class OWLViTModelMeta(BaseModel):
    """OWL-ViT model execution metadata."""
    name: str = Field(default="OWL-ViT", examples=["OWL-ViT"])
    model_id: str = Field(default="google/owlvit-base-patch32", examples=["google/owlvit-base-patch32"])
    parameters: int = Field(..., examples=[153231879])
    device: str = Field(default="cpu", examples=["cpu"])


class OWLViTSummary(BaseModel):
    """Summary of detection run."""
    total_queries: int = Field(..., examples=[3])
    total_detections: int = Field(..., examples=[2])
    score_threshold: float = Field(..., examples=[0.10])
    prompt_template: str = Field(..., examples=["a photo of a {}"])


class OWLViTTiming(BaseModel):
    """Granular inference timing breakdown."""
    preprocessing_ms: float = Field(..., examples=[12.4])
    inference_ms: float = Field(..., examples=[1280.5])
    total_ms: float = Field(..., examples=[1292.9])


class OpenVocabDetectionResponse(BaseModel):
    """Response contract for /api/v1/open-vocabulary/detect."""
    success: bool = Field(default=True, examples=[True])
    mode: str = Field(default="open_vocabulary_detection", examples=["open_vocabulary_detection"])
    image: ImageInfo
    model: OWLViTModelMeta
    summary: OWLViTSummary
    queries: List[OWLViTQueryGroup] = Field(default_factory=list)
    detections: List[OWLViTDetectionItem] = Field(default_factory=list)
    timing: OWLViTTiming


class OpenVocabUnloadResponse(BaseModel):
    """Response contract for /api/v1/open-vocabulary/unload."""
    success: bool = Field(..., examples=[True])
    message: str = Field(..., examples=["OWL-ViT model unloaded from memory."])
    unloaded: bool = Field(..., examples=[True])


# ---------------------------------------------------------------------------
# Region-Level Grounded Recognition Schemas (Phase 6C)
# ---------------------------------------------------------------------------

class RegionDetectionMeta(BaseModel):
    """Detection stage metadata for a grounded region."""
    label: str = Field(..., examples=["dog"])
    score: float = Field(..., examples=[0.7184])
    prompt_used: str = Field(..., examples=["a photo of a dog"])


class RegionCropMeta(BaseModel):
    """Crop boundary and padding specification."""
    width: int = Field(..., examples=[320])
    height: int = Field(..., examples=[240])
    padding_percent: int = Field(default=10, examples=[10])


class RegionCandidateRank(BaseModel):
    """Candidate query ranking for a single cropped region."""
    query: str = Field(..., examples=["golden retriever"])
    similarity_score: float = Field(..., examples=[0.3124])
    rank: int = Field(..., examples=[1])


class RegionRecognitionMeta(BaseModel):
    """Semantic recognition refinement output for a region."""
    top_match: Optional[RegionCandidateRank] = None
    rankings: List[RegionCandidateRank] = Field(default_factory=list)
    total_candidates: int = Field(..., examples=[5])


class RegionItem(BaseModel):
    """Structured representation of a single grounded visual region."""
    region_id: str = Field(..., examples=["reg-01"])
    detection: RegionDetectionMeta
    original_box: OWLViTPixelBox
    padded_box: OWLViTPixelBox
    normalized_box: OWLViTNormalizedBox
    crop: RegionCropMeta
    recognition: Optional[RegionRecognitionMeta] = None


class RegionPipelineSummary(BaseModel):
    """Summary of region recognition execution run."""
    total_queries: int = Field(..., examples=[5])
    detected_regions_count: int = Field(..., examples=[2])
    analyzed_regions_count: int = Field(..., examples=[2])
    detection_threshold: float = Field(..., examples=[0.10])
    crop_padding_percent: int = Field(..., examples=[10])
    batch_size_used: int = Field(..., examples=[2])
    prompt_template: str = Field(..., examples=["a photo of a {}"])


class RegionPipelineModels(BaseModel):
    """Backbone model specifications for two-stage pipeline."""
    detector: str = Field(default="OWL-ViT base patch32", examples=["OWL-ViT base patch32"])
    recognizer: str = Field(default="OpenCLIP-ViT-B-32", examples=["OpenCLIP-ViT-B-32"])
    detector_parameters: int = Field(..., examples=[153231879])
    device: str = Field(default="cpu", examples=["cpu"])


class RegionPipelineTiming(BaseModel):
    """Granular latency profile across two-stage execution."""
    owlvit_detection_ms: float = Field(..., examples=[1280.5])
    crop_preparation_ms: float = Field(..., examples=[0.12])
    openclip_batch_ms: float = Field(..., examples=[525.3])
    total_pipeline_ms: float = Field(..., examples=[1805.9])


class RegionRecognitionResponse(BaseModel):
    """Unified response contract for /api/v1/open-vocabulary/region-recognition."""
    success: bool = Field(default=True, examples=[True])
    mode: str = Field(default="region_recognition", examples=["region_recognition"])
    image: ImageInfo
    summary: RegionPipelineSummary
    models: RegionPipelineModels
    regions: List[RegionItem] = Field(default_factory=list)
    timing: RegionPipelineTiming



# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """Structured error payload."""
    code: str = Field(..., examples=["INVALID_IMAGE"])
    message: str = Field(..., examples=["The uploaded file is not a valid image."])
    request_id: Optional[str] = Field(default=None, examples=["req-7b89f2a0"])


class ErrorResponse(BaseModel):
    """Normalized error envelope returned for all 4xx and 5xx responses."""
    success: bool = Field(default=False, examples=[False])
    error: ErrorDetail

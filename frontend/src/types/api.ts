/**
 * TypeScript Interfaces strictly matching FastAPI Serving Layer schemas.
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  engine: string;
  models: {
    classifier: boolean;
    detector: boolean;
  };
}

export interface ReadyResponse {
  status: string;
  engine: string;
  models_ready: boolean;
  classifier_loaded: boolean;
  detector_loaded: boolean;
}

export interface ClassifierModelInfo {
  model: string;
  vocabulary: string;
  classes: number;
  input_shape: number[];
}

export interface DetectorModelInfo {
  model: string;
  vocabulary: string;
  semantic_categories: number;
  input_format: string;
}

export interface OpenVocabModelInfo {
  model: string;
  pretrained: string;
  supports_text_queries: boolean;
  supports_bounding_boxes: boolean;
  max_queries: number;
  max_query_length: number;
}

export interface OpenVocabDetectionModelInfo {
  model: string;
  model_id: string;
  supports_text_queries: boolean;
  supports_bounding_boxes: boolean;
  loaded: boolean;
  device: string;
  lazy_loaded: boolean;
  max_queries: number;
  max_query_length: number;
}

export interface RegionIntelligenceModelInfo {
  available: boolean;
  detector: string;
  recognizer: string;
  lazy_loaded: boolean;
  max_regions: number;
  crop_padding_percent: number;
}

export interface ModelInfoResponse {
  classifier: ClassifierModelInfo;
  detector: DetectorModelInfo;
  open_vocabulary?: OpenVocabModelInfo;
  open_vocabulary_detection?: OpenVocabDetectionModelInfo;
  region_intelligence?: RegionIntelligenceModelInfo;
  supported_formats: string[];
  supported_modes: string[];
  max_upload_size_mb: number;
}

export interface ImageInfo {
  file_name: string;
  file_path?: string;
  width: number;
  height: number;
  channels: number;
  format: string;
  aspect_ratio: number;
  file_size_bytes: number;
}

export interface PredictionCandidate {
  label: string;
  confidence: number;
  confidence_percent: number;
}

export interface ClassificationResult {
  success: boolean;
  mode: string;
  status: "confident" | "low_confidence" | string;
  top1: PredictionCandidate;
  top5: PredictionCandidate[];
  predictions: PredictionCandidate[];
  confidence_threshold: number;
  is_confident: boolean;
  inference_ms: number;
  preprocessing_ms: number;
  total_ms: number;
  model: string;
  input_dimensions: string;
  error?: string;
  error_code?: string;
}

export interface BoxCoordinates {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
  normalized: [number, number, number, number]; // [ymin, xmin, ymax, xmax] in [0, 1]
}

export interface DetectedObject {
  label: string;
  class_id: number;
  confidence: number;
  confidence_percent: number;
  box: BoxCoordinates;
}

export interface DetectionResult {
  success: boolean;
  mode: string;
  status: "objects_detected" | "no_confident_detections" | string;
  count: number;
  objects: DetectedObject[];
  confidence_threshold: number;
  rendered_image_path?: string | null;
  inference_ms: number;
  preprocessing_ms: number;
  total_ms: number;
  model: string;
  image_dimensions?: {
    width: number;
    height: number;
  };
  error?: string;
  error_code?: string;
}

export interface QueryRankItem {
  query: string;
  similarity_score: number;
  rank: number;
}

export interface OpenVocabResult {
  top_match: QueryRankItem;
  all_ranked: QueryRankItem[];
  total_queries_evaluated: number;
  prompt_template_applied: string;
}

export interface OpenVocabMetadata {
  image_filename: string;
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  device: string;
  model_name: string;
  pretrained_weights: string;
}

export interface OpenVocabResponse {
  success: boolean;
  result: OpenVocabResult;
  metadata: OpenVocabMetadata;
  request_id?: string | null;
}

// ---------------------------------------------------------------------------
// Open-Vocabulary Object Detection (OWL-ViT) Interfaces
// ---------------------------------------------------------------------------

export interface OWLViTPixelBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  width: number;
  height: number;
}

export interface OWLViTNormalizedBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface OWLViTDetectionItem {
  detection_id: string;
  query: string;
  prompt_used: string;
  score: number;
  box: OWLViTPixelBox;
  box_normalized: OWLViTNormalizedBox;
}

export interface OWLViTQueryGroup {
  query: string;
  prompt_applied: string;
  count: number;
  top_score: number;
  detections: OWLViTDetectionItem[];
}

export interface OWLViTModelMeta {
  name: string;
  model_id: string;
  parameters: number;
  device: string;
}

export interface OWLViTSummary {
  total_queries: number;
  total_detections: number;
  score_threshold: number;
  prompt_template: string;
}

export interface OWLViTTiming {
  preprocessing_ms: number;
  inference_ms: number;
  total_ms: number;
}

export interface OpenVocabDetectionResponse {
  success: boolean;
  mode: "open_vocabulary_detection";
  image: ImageInfo;
  model: OWLViTModelMeta;
  summary: OWLViTSummary;
  queries: OWLViTQueryGroup[];
  detections: OWLViTDetectionItem[];
  timing: OWLViTTiming;
}

export interface RegionDetectionMeta {
  label: string;
  score: number;
  prompt_used: string;
}

export interface RegionCropMeta {
  width: number;
  height: number;
  padding_percent: number;
}

export interface RegionCandidateRank {
  query: string;
  similarity_score: number;
  rank: number;
}

export interface RegionRecognitionMeta {
  top_match?: RegionCandidateRank | null;
  rankings: RegionCandidateRank[];
  total_candidates: number;
}

export interface RegionItem {
  region_id: string;
  detection: RegionDetectionMeta;
  original_box: OWLViTPixelBox;
  padded_box: OWLViTPixelBox;
  normalized_box: OWLViTNormalizedBox;
  crop: RegionCropMeta;
  recognition?: RegionRecognitionMeta | null;
}

export interface RegionPipelineSummary {
  total_queries: number;
  detected_regions_count: number;
  analyzed_regions_count: number;
  detection_threshold: number;
  crop_padding_percent: number;
  batch_size_used: number;
  prompt_template: string;
}

export interface RegionPipelineModels {
  detector: string;
  recognizer: string;
  detector_parameters: number;
  device: string;
}

export interface RegionPipelineTiming {
  owlvit_detection_ms: number;
  crop_preparation_ms: number;
  openclip_batch_ms: number;
  total_pipeline_ms: number;
}

export interface RegionRecognitionResponse {
  success: boolean;
  mode: "region_recognition";
  image: ImageInfo;
  summary: RegionPipelineSummary;
  models: RegionPipelineModels;
  regions: RegionItem[];
  timing: RegionPipelineTiming;
}

export interface RecognitionMetadata {
  engine: string;
  classifier_model?: string | null;
  detector_model?: string | null;
  classification_threshold: number;
  detection_threshold: number;
  total_inference_ms: number;
}

export interface RecognitionResponse {
  success: boolean;
  mode: "all" | "classification" | "detection" | "open_vocabulary" | "open_vocabulary_detection" | "region_recognition" | string;
  image: ImageInfo;
  metadata: RecognitionMetadata;
  classification?: ClassificationResult | null;
  detection?: DetectionResult | null;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  request_id?: string | null;
}

export interface ApiErrorResponse {
  success: false;
  error: ApiErrorDetail;
}

export type RecognitionMode =
  | "all"
  | "classification"
  | "detection"
  | "open_vocabulary"
  | "open_vocabulary_detection"
  | "region_recognition";


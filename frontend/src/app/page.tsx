"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Boxes,
  Compass,
  Cpu,
  Layers,
  Loader2,
  RefreshCw,
  Sparkles,
  Tag,
  Upload,
} from "lucide-react";
import { ClassificationCard } from "../components/ClassificationCard";
import { DetectionViewer } from "../components/DetectionViewer";
import { DetectionsList } from "../components/DetectionsList";
import { ErrorAlert } from "../components/ErrorAlert";
import { Header } from "../components/Header";
import { ImageDropzone } from "../components/ImageDropzone";
import { ImagePreview } from "../components/ImagePreview";
import { OpenVocabDetectionPanel } from "../components/OpenVocabDetectionPanel";
import { OpenVocabPanel } from "../components/OpenVocabPanel";
import { RegionIntelligencePanel } from "../components/RegionIntelligencePanel";
import { TechnicalMetadata } from "../components/TechnicalMetadata";
import {
  checkHealth,
  detectOpenVocab,
  getFriendlyErrorMessage,
  recognizeImage,
  recognizeOpenVocab,
  recognizeRegions,
} from "../lib/api";
import {
  DetectedObject,
  HealthResponse,
  OpenVocabDetectionResponse,
  OpenVocabResponse,
  RecognitionMode,
  RecognitionResponse,
  RegionRecognitionResponse,
} from "../types/api";

export default function HomePage() {
  // Application State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [mode, setMode] = useState<RecognitionMode>("all");
  const [classificationThreshold, setClassificationThreshold] = useState<number>(0.4);
  const [detectionThreshold, setDetectionThreshold] = useState<number>(0.4);

  // Open-Vocabulary Recognition (OpenCLIP) State
  const [openVocabQueries, setOpenVocabQueries] = useState<string[]>([
    "rose",
    "sunflower",
    "tulip",
    "automobile",
    "laptop",
  ]);
  const [promptTemplate, setPromptTemplate] = useState<string>("a photo of a {}");
  const [openVocabResult, setOpenVocabResult] = useState<OpenVocabResponse | null>(null);

  // Open-Vocabulary Object Detection (OWL-ViT) State
  const [openVocabDetQueries, setOpenVocabDetQueries] = useState<string[]>([
    "dog",
    "person",
    "car",
    "chair",
    "laptop",
  ]);
  const [openVocabDetPromptTemplate, setOpenVocabDetPromptTemplate] = useState<string>("a photo of a {}");
  const [openVocabDetScoreThreshold, setOpenVocabDetScoreThreshold] = useState<number>(0.10);
  const [openVocabDetTopK, setOpenVocabDetTopK] = useState<number>(20);
  const [openVocabDetResult, setOpenVocabDetResult] = useState<OpenVocabDetectionResponse | null>(null);

  // Region Intelligence (Phase 6C - Two-Stage Grounded Recognition) State
  const [regionTextQueries, setRegionTextQueries] = useState<string>("dog, person, car, chair, laptop");
  const [regionPromptTemplate, setRegionPromptTemplate] = useState<string>("a photo of a {}");
  const [regionDetectionThreshold, setRegionDetectionThreshold] = useState<number>(0.10);
  const [regionCropPaddingPercent, setRegionCropPaddingPercent] = useState<number>(10);
  const [regionMaxRegions, setRegionMaxRegions] = useState<number>(20);
  const [regionResult, setRegionResult] = useState<RegionRecognitionResponse | null>(null);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);

  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [loadingStatusText, setLoadingStatusText] = useState<string>("Analyzing image...");
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState<{
    title: string;
    message: string;
    code?: string;
    requestId?: string | null;
  } | null>(null);

  const [selectedDetectionIndex, setSelectedDetectionIndex] = useState<number | null>(null);

  // Health State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  // Abort Controller Ref for in-flight cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch API Health on load
  const fetchHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  // Clean up Object URL when file changes
  useEffect(() => {
    if (!selectedFile) {
      setPreviewSrc(null);
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewSrc(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [selectedFile]);

  const handleFileSelected = (file: File) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setError(null);
    setResult(null);
    setOpenVocabResult(null);
    setOpenVocabDetResult(null);
    setSelectedFile(file);
    setSelectedDetectionIndex(null);
  };

  const handleRemoveImage = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setSelectedFile(null);
    setResult(null);
    setOpenVocabResult(null);
    setOpenVocabDetResult(null);
    setRegionResult(null);
    setSelectedRegionId(null);
    setError(null);
    setSelectedDetectionIndex(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || analyzing) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setAnalyzing(true);
    setError(null);
    setResult(null);
    setOpenVocabResult(null);
    setOpenVocabDetResult(null);
    setRegionResult(null);
    setSelectedRegionId(null);
    setSelectedDetectionIndex(null);

    try {
      if (mode === "open_vocabulary") {
        if (openVocabQueries.length === 0) {
          setError({
            title: "No Concepts Specified",
            message: "Please enter at least one candidate concept query for Open-Vocabulary recognition.",
            code: "INVALID_TEXT_QUERIES",
          });
          setAnalyzing(false);
          return;
        }

        setLoadingStatusText("Evaluating image with OpenCLIP ViT-B/32...");
        const data = await recognizeOpenVocab(
          selectedFile,
          {
            text_queries: openVocabQueries,
            top_k: 5,
            prompt_template: promptTemplate,
          },
          controller.signal
        );
        setOpenVocabResult(data);
      } else if (mode === "open_vocabulary_detection") {
        if (openVocabDetQueries.length === 0) {
          setError({
            title: "No Target Queries Specified",
            message: "Please enter at least one target object query for Open-Vocabulary Detection.",
            code: "INVALID_TEXT_QUERIES",
          });
          setAnalyzing(false);
          return;
        }

        setLoadingStatusText("Loading OWL-ViT detector & analyzing image...");
        const data = await detectOpenVocab(
          selectedFile,
          {
            text_queries: openVocabDetQueries,
            top_k: openVocabDetTopK,
            score_threshold: openVocabDetScoreThreshold,
            prompt_template: openVocabDetPromptTemplate,
          },
          controller.signal
        );
        setOpenVocabDetResult(data);
      } else if (mode === "region_recognition") {
        if (!regionTextQueries.trim()) {
          setError({
            title: "No Concepts Specified",
            message: "Please enter at least one candidate query for Region-Level Grounded Recognition.",
            code: "INVALID_TEXT_QUERIES",
          });
          setAnalyzing(false);
          return;
        }

        setLoadingStatusText("Grounding & Recognizing Regions (OWL-ViT + OpenCLIP Batch)...");
        const data = await recognizeRegions(
          selectedFile,
          {
            text_queries: regionTextQueries,
            detection_threshold: regionDetectionThreshold,
            crop_padding_percent: regionCropPaddingPercent,
            max_regions: regionMaxRegions,
            prompt_template: regionPromptTemplate,
          },
          controller.signal
        );
        setRegionResult(data);
      } else {
        setLoadingStatusText("Analyzing Image with Neural Networks...");
        const data = await recognizeImage(
          selectedFile,
          {
            mode,
            classification_threshold: classificationThreshold,
            detection_threshold: detectionThreshold,
            top_k: 5,
            render_visualization: false,
          },
          controller.signal
        );
        setResult(data);
      }
    } catch (err: unknown) {
      if ((err as Error)?.name !== "AbortError") {
        const friendly = getFriendlyErrorMessage(err);
        setError(friendly);
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const hasResults = result !== null || openVocabResult !== null || openVocabDetResult !== null || regionResult !== null;

  // Convert OWL-ViT detections to DetectionViewer DetectedObject format
  const mappedOwlVitDetections: DetectedObject[] = openVocabDetResult
    ? openVocabDetResult.detections.map((det, idx) => ({
        label: det.query,
        class_id: idx,
        confidence: det.score,
        confidence_percent: Math.round(det.score * 10000) / 100,
        box: {
          x1: det.box.x_min,
          y1: det.box.y_min,
          x2: det.box.x_max,
          y2: det.box.y_max,
          width: det.box.width,
          height: det.box.height,
          normalized: [
            det.box_normalized.y_min,
            det.box_normalized.x_min,
            det.box_normalized.y_max,
            det.box_normalized.x_max,
          ],
        },
      }))
    : [];

  // Convert Region Intelligence regions to DetectionViewer DetectedObject format
  const mappedRegionDetections: DetectedObject[] = regionResult
    ? regionResult.regions.map((reg, idx) => ({
        label: `${reg.region_id}: ${reg.recognition?.top_match?.query || reg.detection.label}`,
        class_id: idx,
        confidence: reg.recognition?.top_match?.similarity_score ?? reg.detection.score,
        confidence_percent: Math.round((reg.recognition?.top_match?.similarity_score ?? reg.detection.score) * 10000) / 100,
        box: {
          x1: reg.padded_box.x_min,
          y1: reg.padded_box.y_min,
          x2: reg.padded_box.x_max,
          y2: reg.padded_box.y_max,
          width: reg.padded_box.width,
          height: reg.padded_box.height,
          normalized: [
            reg.normalized_box.y_min,
            reg.normalized_box.x_min,
            reg.normalized_box.y_max,
            reg.normalized_box.x_max,
          ],
        },
      }))
    : [];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50/50">
      {/* Top Navigation Header */}
      <Header
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={fetchHealth}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Error Alert if any */}
        {error && (
          <ErrorAlert
            title={error.title}
            message={error.message}
            code={error.code}
            requestId={error.requestId}
            onRetry={handleAnalyze}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Phase A: Upload Zone (when no file is selected) */}
        {!selectedFile && (
          <div className="max-w-3xl mx-auto mt-4 space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                Computer Vision Analysis
              </h2>
              <p className="text-sm text-slate-500 max-w-lg mx-auto">
                Upload any image to classify visual categories, detect entities with bounding boxes,
                or query open-vocabulary concepts using OpenCLIP &amp; Google OWL-ViT.
              </p>
            </div>

            <ImageDropzone onFileSelected={handleFileSelected} />

            {/* Feature highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-4 text-xs text-slate-600">
              <div className="p-3.5 rounded-xl bg-white border border-slate-200/80 flex items-center gap-2.5">
                <Tag className="w-4 h-4 text-sky-500 shrink-0" />
                <span>1,000 ImageNet Classes</span>
              </div>
              <div className="p-3.5 rounded-xl bg-white border border-slate-200/80 flex items-center gap-2.5">
                <Boxes className="w-4 h-4 text-emerald-500 shrink-0" />
                <span>80 COCO Object Boxes</span>
              </div>
              <div className="p-3.5 rounded-xl bg-white border border-slate-200/80 flex items-center gap-2.5">
                <Compass className="w-4 h-4 text-indigo-500 shrink-0" />
                <span>OpenCLIP Zero-Shot</span>
              </div>
              <div className="p-3.5 rounded-xl bg-white border border-slate-200/80 flex items-center gap-2.5">
                <Sparkles className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>OWL-ViT Open Detection</span>
              </div>
            </div>
          </div>
        )}

        {/* Phase B: Image Preview & Settings (when file selected but not yet analyzed) */}
        {selectedFile && !hasResults && (
          <div className="max-w-3xl mx-auto space-y-6">
            <ImagePreview
              file={selectedFile}
              mode={mode}
              onModeChange={setMode}
              classificationThreshold={classificationThreshold}
              onClassificationThresholdChange={setClassificationThreshold}
              detectionThreshold={detectionThreshold}
              onDetectionThresholdChange={setDetectionThreshold}
              onAnalyze={handleAnalyze}
              onChangeImage={handleRemoveImage}
              onRemoveImage={handleRemoveImage}
              analyzing={analyzing}
            />

            {/* Open-Vocabulary Recognition Config Panel (Phase 6A) */}
            {mode === "open_vocabulary" && (
              <OpenVocabPanel
                queries={openVocabQueries}
                onQueriesChange={setOpenVocabQueries}
                promptTemplate={promptTemplate}
                onPromptTemplateChange={setPromptTemplate}
                result={null}
                disabled={analyzing}
              />
            )}

            {/* Open-Vocabulary Detection Config Panel (Phase 6B) */}
            {mode === "open_vocabulary_detection" && (
              <OpenVocabDetectionPanel
                queries={openVocabDetQueries}
                onQueriesChange={setOpenVocabDetQueries}
                promptTemplate={openVocabDetPromptTemplate}
                onPromptTemplateChange={setOpenVocabDetPromptTemplate}
                scoreThreshold={openVocabDetScoreThreshold}
                onScoreThresholdChange={setOpenVocabDetScoreThreshold}
                topK={openVocabDetTopK}
                onTopKChange={setOpenVocabDetTopK}
                result={null}
                disabled={analyzing}
              />
            )}

            {/* Region Intelligence Config Panel (Phase 6C) */}
            {mode === "region_recognition" && (
              <RegionIntelligencePanel
                response={null}
                isLoading={analyzing}
                selectedRegionId={selectedRegionId}
                onSelectRegion={setSelectedRegionId}
                textQueries={regionTextQueries}
                onTextQueriesChange={setRegionTextQueries}
                detectionThreshold={regionDetectionThreshold}
                onDetectionThresholdChange={setRegionDetectionThreshold}
                cropPaddingPercent={regionCropPaddingPercent}
                onCropPaddingPercentChange={setRegionCropPaddingPercent}
                maxRegions={regionMaxRegions}
                onMaxRegionsChange={setRegionMaxRegions}
                promptTemplate={regionPromptTemplate}
                onPromptTemplateChange={setRegionPromptTemplate}
                onRunRecognition={handleAnalyze}
                hasImage={!!selectedFile}
              />
            )}
          </div>
        )}

        {/* Phase C: Standard Results Dashboard (Phases 1-5 Unified / Classification / Detection) */}
        {selectedFile && previewSrc && result && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Top Action Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-50 text-emerald-600 rounded-xl border border-emerald-200">
                  <Sparkles className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    Analysis Completed Successfully
                  </h3>
                  <p className="text-xs text-slate-500">
                    Mode: <span className="font-semibold uppercase text-slate-700">{result.mode}</span> •
                    Total Time: <span className="font-semibold text-slate-700">{result.metadata.total_inference_ms.toFixed(1)} ms</span>
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 font-semibold text-xs transition-colors flex items-center gap-1.5 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${analyzing ? "animate-spin" : ""}`} />
                  <span>Re-analyze</span>
                </button>
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs transition-colors shadow-sm flex items-center gap-1.5"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>New Image</span>
                </button>
              </div>
            </div>

            {/* Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Interactive Image Viewer (7 cols) */}
              <div className="lg:col-span-7 space-y-6">
                <DetectionViewer
                  imageSrc={previewSrc}
                  imageInfo={result.image}
                  detections={result.detection?.objects || []}
                  selectedDetectionIndex={selectedDetectionIndex}
                  onSelectDetection={setSelectedDetectionIndex}
                />
              </div>

              {/* Right Column: Cards (5 cols) */}
              <div className="lg:col-span-5 space-y-6">
                {/* Classification Card if available */}
                {result.classification && (
                  <ClassificationCard classification={result.classification} />
                )}

                {/* Detections List Card if available */}
                {result.detection && (
                  <DetectionsList
                    detection={result.detection}
                    selectedDetectionIndex={selectedDetectionIndex}
                    onSelectDetection={setSelectedDetectionIndex}
                  />
                )}
              </div>
            </div>

            {/* Technical Metadata Drawer */}
            <TechnicalMetadata response={result} />
          </div>
        )}

        {/* Phase D: Open-Vocabulary Recognition Results (Phase 6A) */}
        {selectedFile && previewSrc && openVocabResult && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Top Action Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-50 text-indigo-600 rounded-xl border border-indigo-200">
                  <Compass className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    Open-Vocabulary Analysis Completed
                  </h3>
                  <p className="text-xs text-slate-500">
                    Model: <span className="font-semibold text-slate-700">{openVocabResult.metadata.model_name}</span> •
                    Inference Time: <span className="font-semibold text-slate-700">{openVocabResult.metadata.inference_time_ms.toFixed(1)} ms</span> •
                    Device: <span className="font-semibold text-slate-700 uppercase">{openVocabResult.metadata.device}</span>
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 font-semibold text-xs transition-colors flex items-center gap-1.5 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${analyzing ? "animate-spin" : ""}`} />
                  <span>Re-evaluate</span>
                </button>
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-colors shadow-sm flex items-center gap-1.5"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>New Image</span>
                </button>
              </div>
            </div>

            {/* Open-Vocabulary Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Image Card */}
              <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Input Image Reference
                </h3>
                <div className="w-full h-80 bg-slate-100 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewSrc}
                    alt={selectedFile.name}
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 border-t border-slate-100 pt-3">
                  <span className="font-medium truncate max-w-[200px]">{selectedFile.name}</span>
                  <span>{openVocabResult.metadata.image_width} × {openVocabResult.metadata.image_height} px</span>
                </div>
              </div>

              {/* Right Column: Open-Vocabulary Panel with similarity ranking */}
              <div className="lg:col-span-7 space-y-6">
                <OpenVocabPanel
                  queries={openVocabQueries}
                  onQueriesChange={setOpenVocabQueries}
                  promptTemplate={promptTemplate}
                  onPromptTemplateChange={setPromptTemplate}
                  result={openVocabResult}
                  disabled={analyzing}
                />
              </div>
            </div>
          </div>
        )}

        {/* Phase E: Open-Vocabulary Object Detection Results (Phase 6B - OWL-ViT) */}
        {selectedFile && previewSrc && openVocabDetResult && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Top Action Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-50 text-emerald-600 rounded-xl border border-emerald-200">
                  <Sparkles className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    Open-Vocabulary Detection Completed
                  </h3>
                  <p className="text-xs text-slate-500">
                    Model: <span className="font-semibold text-slate-700">{openVocabDetResult.model.name} ({openVocabDetResult.model.model_id})</span> •
                    Inference Time: <span className="font-semibold text-slate-700">{openVocabDetResult.timing.total_ms.toFixed(1)} ms</span> •
                    Detections: <span className="font-semibold text-slate-700">{openVocabDetResult.summary.total_detections} boxes</span>
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 font-semibold text-xs transition-colors flex items-center gap-1.5 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${analyzing ? "animate-spin" : ""}`} />
                  <span>Re-detect</span>
                </button>
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors shadow-sm flex items-center gap-1.5"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>New Image</span>
                </button>
              </div>
            </div>

            {/* Results Grid with Synchronized Bounding Boxes */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Interactive Image Viewer (7 cols) */}
              <div className="lg:col-span-7 space-y-6">
                <DetectionViewer
                  imageSrc={previewSrc}
                  imageInfo={openVocabDetResult.image}
                  detections={mappedOwlVitDetections}
                  selectedDetectionIndex={selectedDetectionIndex}
                  onSelectDetection={setSelectedDetectionIndex}
                />
              </div>

              {/* Right Column: Open-Vocabulary Detection Breakdown Panel (5 cols) */}
              <div className="lg:col-span-5 space-y-6">
                <OpenVocabDetectionPanel
                  queries={openVocabDetQueries}
                  onQueriesChange={setOpenVocabDetQueries}
                  promptTemplate={openVocabDetPromptTemplate}
                  onPromptTemplateChange={setOpenVocabDetPromptTemplate}
                  scoreThreshold={openVocabDetScoreThreshold}
                  onScoreThresholdChange={setOpenVocabDetScoreThreshold}
                  topK={openVocabDetTopK}
                  onTopKChange={setOpenVocabDetTopK}
                  result={openVocabDetResult}
                  disabled={analyzing}
                />
              </div>
            </div>
          </div>
        )}

        {/* Phase F: Region Intelligence Results (Phase 6C - Grounded Two-Stage Cascade) */}
        {selectedFile && previewSrc && regionResult && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Top Action Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-tr from-cyan-600 to-emerald-600 text-white rounded-xl shadow-sm">
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    Region-Level Grounded Intelligence Completed
                  </h3>
                  <p className="text-xs text-slate-500">
                    Two-Stage Pipeline: <span className="font-semibold text-slate-700">{regionResult.models.detector} &rarr; {regionResult.models.recognizer}</span> •
                    Regions Analyzed: <span className="font-semibold text-emerald-600">{regionResult.regions.length}</span> •
                    Pipeline Time: <span className="font-semibold text-slate-700">{regionResult.timing.total_pipeline_ms.toFixed(1)} ms</span>
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 font-semibold text-xs transition-colors flex items-center gap-1.5 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${analyzing ? "animate-spin" : ""}`} />
                  <span>Re-analyze</span>
                </button>
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-xs transition-colors shadow-sm flex items-center gap-1.5"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>New Image</span>
                </button>
              </div>
            </div>

            {/* Results Grid with Synchronized Bounding Boxes */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Interactive Image Viewer (7 cols) */}
              <div className="lg:col-span-7 space-y-6">
                <DetectionViewer
                  imageSrc={previewSrc}
                  imageInfo={regionResult.image}
                  detections={mappedRegionDetections}
                  selectedDetectionIndex={
                    selectedRegionId !== null
                      ? regionResult.regions.findIndex((r) => r.region_id === selectedRegionId)
                      : null
                  }
                  onSelectDetection={(idx) => {
                    if (idx !== null && regionResult.regions[idx]) {
                      setSelectedRegionId(regionResult.regions[idx].region_id);
                    } else {
                      setSelectedRegionId(null);
                    }
                  }}
                />
              </div>

              {/* Right Column: Region Intelligence Panel (5 cols) */}
              <div className="lg:col-span-5 space-y-6">
                <RegionIntelligencePanel
                  response={regionResult}
                  isLoading={analyzing}
                  selectedRegionId={selectedRegionId}
                  onSelectRegion={setSelectedRegionId}
                  textQueries={regionTextQueries}
                  onTextQueriesChange={setRegionTextQueries}
                  detectionThreshold={regionDetectionThreshold}
                  onDetectionThresholdChange={setRegionDetectionThreshold}
                  cropPaddingPercent={regionCropPaddingPercent}
                  onCropPaddingPercentChange={setRegionCropPaddingPercent}
                  maxRegions={regionMaxRegions}
                  onMaxRegionsChange={setRegionMaxRegions}
                  promptTemplate={regionPromptTemplate}
                  onPromptTemplateChange={setRegionPromptTemplate}
                  onRunRecognition={handleAnalyze}
                  hasImage={true}
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white/60 py-6 text-center text-xs text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>ML Image Recognition Serving System • Phase 6C Grounded Region Intelligence</span>
          <span>FastAPI + Next.js + OWL-ViT Base + OpenCLIP ViT-B/32 + MobileNetV2 + SSD</span>
        </div>
      </footer>
    </div>
  );
}


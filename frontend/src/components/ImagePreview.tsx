"use client";

import React, { useEffect, useState } from "react";
import {
  Boxes,
  Compass,
  Eye,
  Layers,
  Loader2,
  RefreshCw,
  Sliders,
  Sparkles,
  Trash2,
} from "lucide-react";
import { RecognitionMode } from "../types/api";

interface ImagePreviewProps {
  file: File;
  mode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  classificationThreshold: number;
  onClassificationThresholdChange: (val: number) => void;
  detectionThreshold: number;
  onDetectionThresholdChange: (val: number) => void;
  onAnalyze: () => void;
  onChangeImage: () => void;
  onRemoveImage: () => void;
  analyzing: boolean;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  file,
  mode,
  onModeChange,
  classificationThreshold,
  onClassificationThresholdChange,
  detectionThreshold,
  onDetectionThresholdChange,
  onAnalyze,
  onChangeImage,
  onRemoveImage,
  analyzing,
}) => {
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <div className="flex flex-col md:flex-row gap-6 items-start">
        {/* Thumbnail Preview */}
        <div className="relative group w-full md:w-56 h-56 shrink-0 bg-slate-100 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center">
          {previewUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt={file.name}
              className="w-full h-full object-contain"
            />
          )}
        </div>

        {/* Details & Controls */}
        <div className="flex-1 w-full flex flex-col justify-between h-full space-y-5">
          {/* File Meta */}
          <div>
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-slate-900 break-all">
                  {file.name}
                </h3>
                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 mt-1">
                  <span>Size: {formatFileSize(file.size)}</span>
                  {dimensions && (
                    <>
                      <span>•</span>
                      <span>
                        Dimensions: {dimensions.width} × {dimensions.height} px
                      </span>
                    </>
                  )}
                  <span>•</span>
                  <span className="uppercase font-medium text-slate-600">
                    {file.name.split(".").pop()}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={onChangeImage}
                  disabled={analyzing}
                  title="Change Image"
                  className="p-1.5 rounded-lg border border-slate-200 text-slate-500 hover:text-slate-800 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs flex items-center gap-1"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Change</span>
                </button>
                <button
                  type="button"
                  onClick={onRemoveImage}
                  disabled={analyzing}
                  title="Remove Image"
                  className="p-1.5 rounded-lg border border-slate-200 text-rose-500 hover:text-rose-700 hover:bg-rose-50 transition-colors disabled:opacity-50 text-xs flex items-center gap-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Remove</span>
                </button>
              </div>
            </div>
          </div>

          {/* Mode Selector */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
              Select Recognition Mode
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-2.5">
              <button
                type="button"
                onClick={() => onModeChange("all")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "all"
                    ? "border-sky-500 bg-sky-50/50 text-sky-950 font-semibold shadow-sm shadow-sky-500/10"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "all" ? "bg-sky-500 text-white" : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Layers className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">All (Unified)</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    Classify &amp; detect
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => onModeChange("classification")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "classification"
                    ? "border-sky-500 bg-sky-50/50 text-sky-950 font-semibold shadow-sm shadow-sky-500/10"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "classification"
                      ? "bg-sky-500 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Eye className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">Classification</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    ImageNet Top-5
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => onModeChange("detection")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "detection"
                    ? "border-sky-500 bg-sky-50/50 text-sky-950 font-semibold shadow-sm shadow-sky-500/10"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "detection"
                      ? "bg-sky-500 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Boxes className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">Object Detection</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    COCO bounding boxes
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => onModeChange("open_vocabulary")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "open_vocabulary"
                    ? "border-indigo-500 bg-indigo-50/50 text-indigo-950 font-semibold shadow-sm shadow-indigo-500/10"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "open_vocabulary"
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Compass className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">Open Vocab</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    OpenCLIP Zero-Shot
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => onModeChange("open_vocabulary_detection")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "open_vocabulary_detection"
                    ? "border-emerald-500 bg-emerald-50/50 text-emerald-950 font-semibold shadow-sm shadow-emerald-500/10"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "open_vocabulary_detection"
                      ? "bg-emerald-600 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">Open Detection</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    OWL-ViT Localization
                  </div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => onModeChange("region_recognition")}
                disabled={analyzing}
                className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                  mode === "region_recognition"
                    ? "border-teal-500 bg-teal-50/50 text-teal-950 font-semibold shadow-sm shadow-teal-500/10 ring-1 ring-teal-500"
                    : "border-slate-200 hover:border-slate-300 bg-white text-slate-700"
                }`}
              >
                <div
                  className={`p-2 rounded-lg ${
                    mode === "region_recognition"
                      ? "bg-gradient-to-tr from-cyan-600 to-emerald-600 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Layers className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold">Region Intelligence</div>
                  <div className="text-[11px] text-slate-500 font-normal">
                    Grounded Recognition
                  </div>
                </div>
              </button>
            </div>
          </div>

          {/* Advanced Threshold Settings Toggle */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-slate-500 hover:text-slate-800 font-medium flex items-center gap-1.5 transition-colors"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>{showAdvanced ? "Hide" : "Customize"} Confidence Thresholds</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 p-4 bg-slate-50 rounded-xl border border-slate-200 grid grid-cols-1 sm:grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-1">
                {(mode === "all" || mode === "classification") && (
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                      <span>Classification Threshold</span>
                      <span className="text-sky-600 font-bold">
                        {Math.round(classificationThreshold * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={classificationThreshold}
                      onChange={(e) =>
                        onClassificationThresholdChange(parseFloat(e.target.value))
                      }
                      className="w-full accent-sky-500 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                    />
                  </div>
                )}

                {(mode === "all" || mode === "detection") && (
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                      <span>Detection Threshold</span>
                      <span className="text-sky-600 font-bold">
                        {Math.round(detectionThreshold * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={detectionThreshold}
                      onChange={(e) =>
                        onDetectionThresholdChange(parseFloat(e.target.value))
                      }
                      className="w-full accent-sky-500 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Primary Action Button */}
          <div className="pt-2">
            <button
              type="button"
              onClick={onAnalyze}
              disabled={analyzing}
              className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-sky-600 hover:bg-sky-500 active:bg-sky-700 text-white font-semibold text-sm shadow-md shadow-sky-600/20 hover:shadow-lg hover:shadow-sky-600/30 transition-all flex items-center justify-center gap-2.5 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analyzing Image with Neural Networks...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Analyze Image</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

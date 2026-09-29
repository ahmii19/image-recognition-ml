"use client";

import React, { useState } from "react";
import {
  RegionRecognitionResponse,
  RegionItem,
} from "../types/api";

interface RegionIntelligencePanelProps {
  response: RegionRecognitionResponse | null;
  isLoading: boolean;
  selectedRegionId: string | null;
  onSelectRegion: (id: string | null) => void;
  textQueries: string;
  onTextQueriesChange: (queries: string) => void;
  detectionThreshold: number;
  onDetectionThresholdChange: (val: number) => void;
  cropPaddingPercent: number;
  onCropPaddingPercentChange: (val: number) => void;
  maxRegions: number;
  onMaxRegionsChange: (val: number) => void;
  promptTemplate: string;
  onPromptTemplateChange: (val: string) => void;
  onRunRecognition: () => void;
  hasImage: boolean;
}

const PRESET_QUERIES = [
  "dog, person, car, bus, wheel",
  "laptop, coffee cup, smartphone, keyboard",
  "camera, tripod, lens, surveyor",
  "rose, sunflower, tulip, daisy, dandelion",
  "cat, sofa, cushion, animal",
  "red sports car, bicycle, sidewalk",
];

const PROMPT_TEMPLATES = [
  "a photo of a {}",
  "a {}",
  "a close-up photo of a {}",
  "this is a {}",
  "an image of a {}",
];

export const RegionIntelligencePanel: React.FC<RegionIntelligencePanelProps> = ({
  response,
  isLoading,
  selectedRegionId,
  onSelectRegion,
  textQueries,
  onTextQueriesChange,
  detectionThreshold,
  onDetectionThresholdChange,
  cropPaddingPercent,
  onCropPaddingPercentChange,
  maxRegions,
  onMaxRegionsChange,
  promptTemplate,
  onPromptTemplateChange,
  onRunRecognition,
  hasImage,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [expandedDetails, setExpandedDetails] = useState<Record<string, boolean>>({});

  const toggleDetails = (regionId: string) => {
    setExpandedDetails((prev) => ({
      ...prev,
      [regionId]: !prev[regionId],
    }));
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.28) return "bg-emerald-500 text-white";
    if (score >= 0.20) return "bg-teal-500 text-white";
    if (score >= 0.12) return "bg-amber-500 text-white";
    return "bg-slate-500 text-white";
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Control Header Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-md">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-emerald-500 flex items-center justify-center text-white font-bold shadow-lg shadow-emerald-500/20">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-bold text-white tracking-wide">
                Region-Level Grounded Intelligence
              </h3>
              <p className="text-xs text-slate-400">
                OWL-ViT Localization &rarr; 10% Symmetrical Context Crop &rarr; Batched OpenCLIP Recognition
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700/60 self-start sm:self-auto"
          >
            {showAdvanced ? "Hide Advanced Settings" : "Configure Parameters"}
          </button>
        </div>

        {/* Text Queries Input */}
        <div className="flex flex-col gap-2 mb-4">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Candidate Concepts / Text Queries
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={textQueries}
              onChange={(e) => onTextQueriesChange(e.target.value)}
              placeholder="e.g. dog, person, sports car, laptop, coffee cup"
              className="flex-1 bg-slate-950/80 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all font-mono"
            />
            <button
              onClick={onRunRecognition}
              disabled={isLoading || !hasImage || !textQueries.trim()}
              className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm shadow-lg shadow-emerald-600/30 transition-all flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Processing Regions...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Analyze Regions</span>
                </>
              )}
            </button>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-1.5 mt-2">
            <span className="text-[11px] text-slate-500 uppercase font-medium mr-1">Presets:</span>
            {PRESET_QUERIES.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onTextQueriesChange(preset)}
                className="text-[11px] bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700/50 transition-colors"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Parameter Controls */}
        {showAdvanced && (
          <div className="pt-4 mt-4 border-t border-slate-800 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 bg-slate-950/40 p-4 rounded-xl">
            {/* Prompt Template */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-slate-300 font-medium">Prompt Template</label>
              <select
                value={promptTemplate}
                onChange={(e) => onPromptTemplateChange(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
              >
                {PROMPT_TEMPLATES.map((tmpl) => (
                  <option key={tmpl} value={tmpl}>
                    {tmpl}
                  </option>
                ))}
              </select>
            </div>

            {/* Context Padding */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Context Padding</span>
                <span className="text-emerald-400 font-mono font-bold">{cropPaddingPercent}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="30"
                step="5"
                value={cropPaddingPercent}
                onChange={(e) => onCropPaddingPercentChange(Number(e.target.value))}
                className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Benchmark recommended: 10%</span>
            </div>

            {/* Detection Cutoff */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Detection Cutoff</span>
                <span className="text-cyan-400 font-mono font-bold">{detectionThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.50"
                step="0.01"
                value={detectionThreshold}
                onChange={(e) => onDetectionThresholdChange(Number(e.target.value))}
                className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Filters low-confidence boxes</span>
            </div>

            {/* Max Regions */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Max Regions</span>
                <span className="text-purple-400 font-mono font-bold">{maxRegions}</span>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={maxRegions}
                onChange={(e) => onMaxRegionsChange(Number(e.target.value))}
                className="w-full accent-purple-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Deterministic selection</span>
            </div>
          </div>
        )}
      </div>

      {/* Results Section */}
      {response && (
        <div className="flex flex-col gap-6">
          {/* Summary / Telemetry Ribbon */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Grounded Regions</span>
              <span className="text-xl font-bold text-emerald-400 font-mono mt-1">
                {response.regions.length}
              </span>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Context Padding</span>
              <span className="text-xl font-bold text-cyan-400 font-mono mt-1">
                {response.summary.crop_padding_percent}%
              </span>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Batch Size</span>
              <span className="text-xl font-bold text-purple-400 font-mono mt-1">
                {response.summary.batch_size_used}
              </span>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">OWL-ViT Stage</span>
              <span className="text-xl font-bold text-slate-200 font-mono mt-1">
                {response.timing.owlvit_detection_ms.toFixed(0)} <span className="text-xs text-slate-400 font-normal">ms</span>
              </span>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">OpenCLIP Batch</span>
              <span className="text-xl font-bold text-slate-200 font-mono mt-1">
                {response.timing.openclip_batch_ms.toFixed(0)} <span className="text-xs text-slate-400 font-normal">ms</span>
              </span>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Total Latency</span>
              <span className="text-xl font-bold text-emerald-400 font-mono mt-1">
                {response.timing.total_pipeline_ms.toFixed(0)} <span className="text-xs text-slate-400 font-normal">ms</span>
              </span>
            </div>
          </div>

          {/* Region Cards Grid */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-bold uppercase tracking-wider text-slate-300">
                Analyzed Visual Regions ({response.regions.length})
              </h4>
              <span className="text-xs text-slate-400">
                Click any region card to focus its bounding box
              </span>
            </div>

            {response.regions.length === 0 ? (
              <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl p-8 text-center text-slate-400 text-sm">
                No visual regions exceeded detection cutoff ({response.summary.detection_threshold}). Try lowering the detection cutoff slider or adding different candidate queries.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {response.regions.map((region: RegionItem) => {
                  const isSelected = selectedRegionId === region.region_id;
                  const topMatch = region.recognition?.top_match;
                  const score = topMatch?.similarity_score ?? 0;
                  const isExpanded = !!expandedDetails[region.region_id];

                  return (
                    <div
                      key={region.region_id}
                      onClick={() => onSelectRegion(isSelected ? null : region.region_id)}
                      className={`cursor-pointer rounded-2xl p-5 transition-all duration-200 border flex flex-col gap-4 ${
                        isSelected
                          ? "bg-emerald-950/40 border-emerald-500 shadow-lg shadow-emerald-500/10 ring-2 ring-emerald-500/30"
                          : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-900"
                      }`}
                    >
                      {/* Region Header */}
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-200 text-xs font-mono font-bold border border-slate-700">
                            {region.region_id}
                          </span>
                          <span className="text-xs text-slate-400 font-medium">
                            Detected: <strong className="text-slate-200">{region.detection.label}</strong> ({(region.detection.score * 100).toFixed(1)}%)
                          </span>
                        </div>

                        <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800/80 text-slate-400 border border-slate-700/60 font-mono">
                          {region.crop.width}&times;{region.crop.height} px (+{region.crop.padding_percent}%)
                        </span>
                      </div>

                      {/* Primary OpenCLIP Refinement Output */}
                      <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800/80 flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                            OpenCLIP Refinement
                          </span>
                          {topMatch && (
                            <span className={`text-[11px] font-mono px-2 py-0.5 rounded font-bold ${getScoreColor(score)}`}>
                              Score: {score.toFixed(4)}
                            </span>
                          )}
                        </div>

                        <div className="text-base font-bold text-white tracking-wide">
                          {topMatch ? topMatch.query : "No confident match"}
                        </div>

                        {/* Visual Cosine Score Meter */}
                        {topMatch && (
                          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-1">
                            <div
                              className="bg-gradient-to-r from-teal-500 to-emerald-400 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${Math.min(100, Math.max(0, (score / 0.40) * 100))}%` }}
                            ></div>
                          </div>
                        )}
                      </div>

                      {/* Top-K Ranked Queries */}
                      {region.recognition && region.recognition.rankings.length > 1 && (
                        <div className="flex flex-col gap-1.5">
                          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                            Candidate Rankings
                          </span>
                          <div className="flex flex-col gap-1 bg-slate-950/40 rounded-lg p-2 border border-slate-800/50">
                            {region.recognition.rankings.slice(0, 4).map((rank) => (
                              <div key={rank.rank} className="flex items-center justify-between text-xs py-0.5">
                                <span className="text-slate-300 font-medium">
                                  #{rank.rank} {rank.query}
                                </span>
                                <span className="text-slate-400 font-mono text-[11px]">
                                  {rank.similarity_score.toFixed(4)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Expandable Technical Coordinates */}
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleDetails(region.region_id);
                        }}
                        className="text-[11px] text-slate-500 hover:text-slate-300 flex items-center justify-between pt-1 border-t border-slate-800/60"
                      >
                        <span>{isExpanded ? "Hide Box Coordinates" : "View Box Coordinates"}</span>
                        <span>{isExpanded ? "\u25B2" : "\u25BC"}</span>
                      </button>

                      {isExpanded && (
                        <div className="text-[11px] font-mono text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800 flex flex-col gap-1">
                          <div>Original: [{region.original_box.x_min}, {region.original_box.y_min}, {region.original_box.x_max}, {region.original_box.y_max}]</div>
                          <div>Padded: [{region.padded_box.x_min}, {region.padded_box.y_min}, {region.padded_box.x_max}, {region.padded_box.y_max}]</div>
                          <div>Normalized: [{region.normalized_box.x_min.toFixed(3)}, {region.normalized_box.y_min.toFixed(3)}, {region.normalized_box.x_max.toFixed(3)}, {region.normalized_box.y_max.toFixed(3)}]</div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

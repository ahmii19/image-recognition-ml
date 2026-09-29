"use client";

import React, { useState } from "react";
import {
  AlertCircle,
  Award,
  Compass,
  FileText,
  Info,
  Plus,
  RotateCcw,
  Sparkles,
  Tag,
  X,
} from "lucide-react";
import { OpenVocabResponse } from "../types/api";

const PRESET_BUNDLES: Record<string, string[]> = {
  "Nature & Flowers": [
    "rose",
    "sunflower",
    "tulip",
    "daisy",
    "dandelion",
    "tree",
    "green grass",
  ],
  "Everyday Objects": [
    "laptop",
    "coffee mug",
    "smartphone",
    "backpack",
    "water bottle",
    "mechanical keyboard",
  ],
  "Animals": [
    "dog",
    "cat",
    "bird",
    "horse",
    "elephant",
    "butterfly",
    "lion",
  ],
  "Vehicles": [
    "sports car",
    "bicycle",
    "motorcycle",
    "airplane",
    "passenger bus",
    "sailboat",
  ],
  "Fine-Grained Floral": [
    "red rose with petals",
    "yellow blooming sunflower",
    "pink carnation",
    "white water lily",
    "purple violet flower",
  ],
};

interface OpenVocabPanelProps {
  queries: string[];
  onQueriesChange: (queries: string[]) => void;
  promptTemplate: string;
  onPromptTemplateChange: (template: string) => void;
  result: OpenVocabResponse | null;
  disabled?: boolean;
}

export const OpenVocabPanel: React.FC<OpenVocabPanelProps> = ({
  queries,
  onQueriesChange,
  promptTemplate,
  onPromptTemplateChange,
  result,
  disabled = false,
}) => {
  const [inputTag, setInputTag] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);

  const handleAddTag = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = inputTag.trim();
    if (!trimmed) return;

    if (queries.includes(trimmed)) {
      setInputError(`"${trimmed}" is already in your candidate list.`);
      return;
    }

    if (queries.length >= 20) {
      setInputError("Maximum 20 candidate queries allowed.");
      return;
    }

    if (trimmed.length > 128) {
      setInputError("Candidate query exceeds 128 character limit.");
      return;
    }

    setInputError(null);
    onQueriesChange([...queries, trimmed]);
    setInputTag("");
  };

  const handleRemoveTag = (tagToRemove: string) => {
    if (disabled) return;
    onQueriesChange(queries.filter((t) => t !== tagToRemove));
  };

  const handleApplyPreset = (presetName: string) => {
    if (disabled) return;
    setInputError(null);
    onQueriesChange(PRESET_BUNDLES[presetName]);
  };

  const handleClearTags = () => {
    if (disabled) return;
    setInputError(null);
    onQueriesChange([]);
  };

  // Convert cosine similarity score (typically ~0.05 to 0.40 for OpenCLIP) into relative bar width
  const calculateBarWidth = (score: number, maxScore: number): number => {
    if (maxScore <= 0) return Math.max(5, Math.min(100, (score + 1) * 50));
    // Scale relative to top match score for distinct visualization
    const normalized = Math.max(0.05, score / maxScore);
    return Math.min(100, Math.round(normalized * 100));
  };

  const topMatch = result?.result.top_match;
  const topScore = topMatch?.similarity_score ?? 1.0;

  return (
    <div className="space-y-6">
      {/* 1. Concepts Input & Configuration Card */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
          <div className="flex items-center gap-2 text-slate-800 font-bold text-base">
            <Compass className="w-5 h-5 text-indigo-600" />
            <span>Open-Vocabulary Concepts</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
              OpenCLIP ViT-B/32
            </span>
            <span className="text-xs text-slate-500 font-medium">
              {queries.length}/20 concepts
            </span>
          </div>
        </div>

        {/* Preset Category Quick Buttons */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Quick Preset Concept Bundles
          </label>
          <div className="flex flex-wrap gap-1.5">
            {Object.keys(PRESET_BUNDLES).map((preset) => (
              <button
                key={preset}
                type="button"
                disabled={disabled}
                onClick={() => handleApplyPreset(preset)}
                className="px-2.5 py-1 rounded-lg border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/50 text-slate-600 hover:text-indigo-700 text-xs font-medium transition-colors disabled:opacity-50"
              >
                {preset}
              </button>
            ))}
            {queries.length > 0 && (
              <button
                type="button"
                disabled={disabled}
                onClick={handleClearTags}
                className="px-2 py-1 rounded-lg border border-rose-200 hover:bg-rose-50 text-rose-600 text-xs font-medium transition-colors disabled:opacity-50 flex items-center gap-1"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Clear</span>
              </button>
            )}
          </div>
        </div>

        {/* Tag Input Form */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Custom Candidate Concepts / Descriptions
          </label>
          <form onSubmit={handleAddTag} className="flex gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={inputTag}
                onChange={(e) => {
                  setInputTag(e.target.value);
                  if (inputError) setInputError(null);
                }}
                disabled={disabled || queries.length >= 20}
                placeholder="Type a concept (e.g. 'golden retriever', 'red sports car') and press Enter"
                className="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 text-xs text-slate-800 placeholder-slate-400 outline-none transition-all disabled:bg-slate-50"
              />
            </div>
            <button
              type="submit"
              disabled={disabled || !inputTag.trim() || queries.length >= 20}
              className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-colors shadow-sm disabled:opacity-50 flex items-center gap-1 shrink-0"
            >
              <Plus className="w-4 h-4" />
              <span>Add</span>
            </button>
          </form>

          {inputError && (
            <div className="mt-2 text-xs text-rose-600 flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{inputError}</span>
            </div>
          )}
        </div>

        {/* Active Concept Chips */}
        <div>
          <div className="flex flex-wrap gap-2 min-h-[44px] p-3 rounded-xl bg-slate-50 border border-slate-100 items-center">
            {queries.length === 0 ? (
              <span className="text-xs text-slate-400 italic">
                No candidate concepts added yet. Choose a preset bundle or add your own terms above.
              </span>
            ) : (
              queries.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-slate-800 text-xs font-semibold shadow-xs"
                >
                  <Tag className="w-3 h-3 text-indigo-500 shrink-0" />
                  <span>{tag}</span>
                  {!disabled && (
                    <button
                      type="button"
                      onClick={() => handleRemoveTag(tag)}
                      className="text-slate-400 hover:text-rose-500 rounded p-0.5 transition-colors"
                      title={`Remove ${tag}`}
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </span>
              ))
            )}
          </div>
        </div>

        {/* Prompt Template Customizer */}
        <div className="pt-2 border-t border-slate-100">
          <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
            <span className="font-semibold flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-slate-400" />
              Vision-Language Prompt Template
            </span>
            <span className="text-[11px] text-slate-400">Must include &quot;&#123;&#125;&quot;</span>
          </div>
          <input
            type="text"
            value={promptTemplate}
            onChange={(e) => onPromptTemplateChange(e.target.value)}
            disabled={disabled}
            placeholder="a photo of a {}"
            className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono text-slate-700 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 disabled:bg-slate-50"
          />
        </div>
      </div>

      {/* 2. Results Section (if inference response exists) */}
      {result && (
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-5 animate-in fade-in duration-300">
          <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2 text-slate-800 font-bold text-base">
              <Sparkles className="w-5 h-5 text-indigo-600" />
              <span>Zero-Shot Similarity Rankings</span>
            </div>
            <span className="text-xs text-slate-500 font-medium">
              {result.result.total_queries_evaluated} concepts evaluated
            </span>
          </div>

          {/* Top-1 Match Spotlight */}
          {topMatch && (
            <div className="p-4 rounded-xl border bg-indigo-50/60 border-indigo-200 shadow-sm">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-indigo-800 mb-1">
                <span className="flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-indigo-600" />
                  Primary Open-Vocabulary Match (#1)
                </span>
                <span className="font-mono font-bold text-indigo-900 text-sm">
                  {topMatch.similarity_score.toFixed(4)}
                </span>
              </div>

              <div className="text-xl sm:text-2xl font-black text-slate-900 tracking-tight my-1 capitalize">
                {topMatch.query}
              </div>

              {/* Similarity meter */}
              <div className="w-full bg-slate-200/80 h-2.5 rounded-full overflow-hidden mt-3">
                <div
                  className="h-full rounded-full bg-indigo-600 transition-all duration-500"
                  style={{ width: "100%" }}
                />
              </div>
              <div className="flex justify-between items-center text-[11px] text-indigo-700 mt-1.5 font-medium">
                <span>Cosine Similarity</span>
                <span>Highest semantic affinity</span>
              </div>
            </div>
          )}

          {/* Full Ranked Similarity Meters */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
              Candidate Similarity Ranking
            </h4>

            <div className="space-y-2.5">
              {result.result.all_ranked.map((item) => {
                const widthPercent = calculateBarWidth(item.similarity_score, topScore);
                const isTop1 = item.rank === 1;

                return (
                  <div
                    key={`${item.query}-${item.rank}`}
                    className={`p-3 rounded-xl border transition-all ${
                      isTop1
                        ? "bg-indigo-50/40 border-indigo-200"
                        : "bg-slate-50/70 border-slate-100"
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-medium mb-1.5">
                      <span className="text-slate-800 flex items-center gap-2 font-semibold capitalize">
                        <span
                          className={`w-5 h-5 rounded-full text-[10px] flex items-center justify-center font-bold ${
                            isTop1
                              ? "bg-indigo-600 text-white"
                              : "bg-slate-200 text-slate-600"
                          }`}
                        >
                          {item.rank}
                        </span>
                        <span>{item.query}</span>
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-slate-400 font-medium">Similarity:</span>
                        <span
                          className={`font-mono font-bold text-xs ${
                            isTop1 ? "text-indigo-700" : "text-slate-700"
                          }`}
                        >
                          {item.similarity_score.toFixed(4)}
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar Meter */}
                    <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isTop1 ? "bg-indigo-600" : "bg-indigo-400/80"
                        }`}
                        style={{ width: `${widthPercent}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Metric Notice */}
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-600 text-xs flex items-start gap-2">
            <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
            <div className="text-[11px] leading-relaxed">
              <span className="font-semibold text-slate-700">Cosine Similarity Metric:</span> Scores
              represent dot products of normalized 512-dimensional image and text unit embeddings
              (e_img &middot; e_txt). Unlike closed-set softmax probabilities, similarity
              scores reflect semantic alignment without bounding box localization.
            </div>
          </div>

          {/* Timing Footer */}
          <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
            <span>Vision-Language Inference Time</span>
            <span className="font-semibold text-slate-700 font-mono">
              {result.metadata.inference_time_ms.toFixed(1)} ms
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

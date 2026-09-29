"use client";

import React, { useState } from "react";
import {
  Boxes,
  Compass,
  Info,
  Layers,
  Plus,
  Sliders,
  Sparkles,
  Tag,
  Trash2,
  X,
} from "lucide-react";
import { OpenVocabDetectionResponse } from "../types/api";

interface OpenVocabDetectionPanelProps {
  queries: string[];
  onQueriesChange: (queries: string[]) => void;
  promptTemplate: string;
  onPromptTemplateChange: (val: string) => void;
  scoreThreshold: number;
  onScoreThresholdChange: (val: number) => void;
  topK: number;
  onTopKChange: (val: number) => void;
  result: OpenVocabDetectionResponse | null;
  disabled?: boolean;
}

const PRESET_COLLECTIONS: { name: string; icon: string; items: string[] }[] = [
  {
    name: "Everyday Objects",
    icon: "📦",
    items: ["dog", "cat", "person", "car", "phone", "laptop", "coffee cup", "chair", "backpack"],
  },
  {
    name: "Nature & Outdoors",
    icon: "🌲",
    items: ["tree", "flower", "dog", "bird", "grass", "mountain"],
  },
  {
    name: "Road & Urban Scene",
    icon: "🚦",
    items: ["car", "bus", "person", "bicycle", "motorcycle", "traffic light", "street"],
  },
];

export const OpenVocabDetectionPanel: React.FC<OpenVocabDetectionPanelProps> = ({
  queries,
  onQueriesChange,
  promptTemplate,
  onPromptTemplateChange,
  scoreThreshold,
  onScoreThresholdChange,
  topK,
  onTopKChange,
  result,
  disabled = false,
}) => {
  const [inputValue, setInputValue] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const handleAddQuery = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    // Handle comma-separated bulk inputs
    const parts = trimmed
      .split(",")
      .map((p) => p.trim())
      .filter((p) => p.length > 0 && p.length <= 128);

    const merged = Array.from(new Set([...queries, ...parts])).slice(0, 20);
    onQueriesChange(merged);
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddQuery();
    }
  };

  const handleRemoveQuery = (indexToRemove: number) => {
    onQueriesChange(queries.filter((_, idx) => idx !== indexToRemove));
  };

  const handleClearAll = () => {
    onQueriesChange([]);
  };

  const handleApplyPreset = (presetItems: string[]) => {
    onQueriesChange(presetItems.slice(0, 20));
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-50 text-emerald-600 rounded-xl border border-emerald-200">
            <Boxes className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Open-Vocabulary Object Detection
            </h3>
            <p className="text-xs text-slate-500">
              Detect and localize arbitrary custom entities using Google OWL-ViT Base.
            </p>
          </div>
        </div>

        {queries.length > 0 && !disabled && (
          <button
            type="button"
            onClick={handleClearAll}
            className="text-xs text-rose-500 hover:text-rose-700 font-medium flex items-center gap-1 transition-colors self-start sm:self-auto"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Queries</span>
          </button>
        )}
      </div>

      {/* Preset Suggestions */}
      {!disabled && (
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
            Suggested Query Presets
          </label>
          <div className="flex flex-wrap gap-2">
            {PRESET_COLLECTIONS.map((preset) => (
              <button
                key={preset.name}
                type="button"
                onClick={() => handleApplyPreset(preset.items)}
                className="px-3 py-1.5 rounded-lg border border-slate-200 hover:border-slate-300 bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-medium transition-colors flex items-center gap-1.5"
              >
                <span>{preset.icon}</span>
                <span>{preset.name}</span>
                <span className="text-[10px] text-slate-400">({preset.items.length})</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Query Input Box */}
      {!disabled && (
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs">
            <label className="font-semibold uppercase tracking-wider text-slate-500">
              Target Object Queries
            </label>
            <span className="text-slate-400">
              {queries.length}/20 queries
            </span>
          </div>

          <div className="flex gap-2">
            <div className="relative flex-1">
              <Tag className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter custom query (e.g. dog, red car, coffee cup)..."
                disabled={disabled || queries.length >= 20}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-xs text-slate-800 placeholder-slate-400 disabled:bg-slate-50 disabled:cursor-not-allowed"
              />
            </div>
            <button
              type="button"
              onClick={handleAddQuery}
              disabled={disabled || !inputValue.trim() || queries.length >= 20}
              className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-semibold text-xs transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              <Plus className="w-4 h-4" />
              <span>Add</span>
            </button>
          </div>
        </div>
      )}

      {/* Active Query Chips */}
      {queries.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {queries.map((query, idx) => (
            <span
              key={`${query}-${idx}`}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/80"
            >
              <span>{query}</span>
              {!disabled && (
                <button
                  type="button"
                  onClick={() => handleRemoveQuery(idx)}
                  className="hover:text-emerald-900 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </span>
          ))}
        </div>
      )}

      {/* Advanced Settings Toggle */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-slate-500 hover:text-slate-800 font-medium flex items-center gap-1.5 transition-colors"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>{showAdvanced ? "Hide" : "Customize"} Detection Threshold &amp; Prompt Template</span>
        </button>

        {showAdvanced && (
          <div className="mt-3 p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-4 animate-in fade-in slide-in-from-top-1">
            {/* Prompt Template */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-700">Prompt Ensembling Template</span>
                <span className="text-slate-400 font-mono text-[11px]">{promptTemplate}</span>
              </div>
              <input
                type="text"
                value={promptTemplate}
                onChange={(e) => onPromptTemplateChange(e.target.value)}
                disabled={disabled}
                placeholder="a photo of a {}"
                className="w-full px-3 py-2 rounded-lg border border-slate-200 bg-white text-xs font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
              <p className="text-[11px] text-slate-500 flex items-center gap-1">
                <Info className="w-3 h-3 text-slate-400 shrink-0" />
                <span>Must include {'"{}"'} which will be populated by each concept query.</span>
              </p>
            </div>

            {/* Threshold & Top-K Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-200/80">
              <div>
                <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                  <span>Score Threshold</span>
                  <span className="text-emerald-600 font-bold font-mono">
                    {scoreThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.90"
                  step="0.05"
                  value={scoreThreshold}
                  onChange={(e) => onScoreThresholdChange(parseFloat(e.target.value))}
                  disabled={disabled}
                  className="w-full accent-emerald-500 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                  <span>Max Detections (Top-K)</span>
                  <span className="text-emerald-600 font-bold font-mono">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  step="1"
                  value={topK}
                  onChange={(e) => onTopKChange(parseInt(e.target.value, 10))}
                  disabled={disabled}
                  className="w-full accent-emerald-500 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results Query Breakdown (if results present) */}
      {result && (
        <div className="space-y-3 pt-4 border-t border-slate-100">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Detections by Query
            </h4>
            <span className="text-xs font-semibold text-emerald-600">
              {result.summary.total_detections} Total Bounding Boxes
            </span>
          </div>

          <div className="space-y-2">
            {result.queries.map((qGroup) => (
              <div
                key={qGroup.query}
                className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-800">{qGroup.query}</span>
                  <span className="text-[11px] text-slate-400">
                    ({qGroup.prompt_applied})
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold ${
                    qGroup.count > 0 ? "bg-emerald-100 text-emerald-800" : "bg-slate-200 text-slate-600"
                  }`}>
                    {qGroup.count} found
                  </span>
                  {qGroup.count > 0 && (
                    <span className="font-mono text-slate-600 text-[11px]">
                      Top: {(qGroup.top_score * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

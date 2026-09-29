"use client";

import React from "react";
import { Box, Boxes, CheckCircle, Info, Layers } from "lucide-react";
import { DetectionResult } from "../types/api";

interface DetectionsListProps {
  detection: DetectionResult;
  selectedDetectionIndex: number | null;
  onSelectDetection: (index: number | null) => void;
}

const BADGE_COLORS = [
  "bg-sky-100 text-sky-800 border-sky-300",
  "bg-emerald-100 text-emerald-800 border-emerald-300",
  "bg-purple-100 text-purple-800 border-purple-300",
  "bg-amber-100 text-amber-800 border-amber-300",
  "bg-rose-100 text-rose-800 border-rose-300",
  "bg-cyan-100 text-cyan-800 border-cyan-300",
  "bg-indigo-100 text-indigo-800 border-indigo-300",
];

export const DetectionsList: React.FC<DetectionsListProps> = ({
  detection,
  selectedDetectionIndex,
  onSelectDetection,
}) => {
  const objects = detection.objects;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2 text-slate-800 font-bold text-base">
            <Boxes className="w-5 h-5 text-sky-600" />
            <span>Detected Entities (COCO-80)</span>
          </div>
          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            {detection.model}
          </span>
        </div>

        {/* Count summary banner */}
        <div className="p-3 bg-slate-50 border border-slate-100 rounded-xl mb-4 flex items-center justify-between text-xs">
          <span className="text-slate-600 font-medium">Objects Detected:</span>
          <span className="font-bold text-slate-900 px-2 py-0.5 rounded bg-white border border-slate-200">
            {detection.count} {detection.count === 1 ? "entity" : "entities"}
          </span>
        </div>

        {/* Objects List */}
        {objects.length === 0 ? (
          <div className="p-8 text-center bg-slate-50/50 rounded-xl border border-dashed border-slate-200">
            <Info className="w-8 h-8 mx-auto text-slate-400 mb-2" />
            <h4 className="text-xs font-bold text-slate-700">No Confident Objects Detected</h4>
            <p className="text-[11px] text-slate-500 mt-1 max-w-xs mx-auto">
              No objects met the {Math.round(detection.confidence_threshold * 100)}% detection
              confidence cutoff. Try lowering the threshold in settings.
            </p>
          </div>
        ) : (
          <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
            {objects.map((item, idx) => {
              const isSelected = selectedDetectionIndex === idx;
              const colorClass = BADGE_COLORS[idx % BADGE_COLORS.length];

              return (
                <div
                  key={`${item.label}-${idx}`}
                  onMouseEnter={() => onSelectDetection(idx)}
                  onMouseLeave={() => onSelectDetection(null)}
                  onClick={() => onSelectDetection(isSelected ? null : idx)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? "border-sky-500 bg-sky-50/80 shadow-md ring-2 ring-sky-500/20 scale-[1.01]"
                      : "border-slate-200 bg-slate-50/40 hover:bg-slate-50 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${colorClass}`}
                      >
                        #{idx + 1}
                      </span>
                      <span className="text-sm font-bold capitalize text-slate-900">
                        {item.label}
                      </span>
                    </div>

                    <span className="text-xs font-bold text-sky-700 bg-sky-100/80 px-2 py-0.5 rounded-md">
                      {item.confidence_percent.toFixed(1)}%
                    </span>
                  </div>

                  {/* Bounding box pixel bounds */}
                  <div className="flex items-center gap-3 text-[11px] text-slate-500 mt-2">
                    <span className="flex items-center gap-1 font-mono text-[10px]">
                      <Box className="w-3 h-3 text-slate-400" />
                      {item.box.width}×{item.box.height}px
                    </span>
                    <span>•</span>
                    <span className="font-mono text-[10px]">
                      Pos: [{item.box.x1}, {item.box.y1}]
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer Timing */}
      <div className="mt-5 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span>Inference Time</span>
        <span className="font-semibold text-slate-600">
          {detection.inference_ms.toFixed(1)} ms
        </span>
      </div>
    </div>
  );
};

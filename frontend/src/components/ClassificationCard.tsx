"use client";

import React from "react";
import { AlertTriangle, Award, CheckCircle2, Sparkles, Tag } from "lucide-react";
import { ClassificationResult } from "../types/api";

interface ClassificationCardProps {
  classification: ClassificationResult;
}

export const ClassificationCard: React.FC<ClassificationCardProps> = ({
  classification,
}) => {
  const top1 = classification.top1;
  const isLowConfidence = classification.status === "low_confidence" || !classification.is_confident;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2 text-slate-800 font-bold text-base">
            <Tag className="w-5 h-5 text-sky-600" />
            <span>Classification (ImageNet)</span>
          </div>
          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            {classification.model}
          </span>
        </div>

        {/* Low confidence warning if applicable */}
        {isLowConfidence && (
          <div className="mb-4 flex items-start gap-2.5 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
            <div>
              <span className="font-semibold">Low Confidence Result:</span> The model is
              uncertain about this classification (under{" "}
              {Math.round(classification.confidence_threshold * 100)}% threshold).
            </div>
          </div>
        )}

        {/* Top 1 Primary Banner */}
        <div
          className={`p-4 rounded-xl border mb-5 transition-all ${
            isLowConfidence
              ? "bg-amber-50/40 border-amber-200"
              : "bg-sky-50/50 border-sky-200 shadow-sm"
          }`}
        >
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
            <span className="flex items-center gap-1.5">
              <Award className="w-4 h-4 text-sky-600" />
              Primary Prediction (Top-1)
            </span>
            <span
              className={`font-bold ${
                isLowConfidence ? "text-amber-700" : "text-sky-700"
              }`}
            >
              {top1.confidence_percent.toFixed(1)}% Match
            </span>
          </div>

          <div className="text-xl sm:text-2xl font-black capitalize text-slate-900 tracking-tight">
            {top1.label}
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-200/80 h-2 rounded-full overflow-hidden mt-3">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isLowConfidence ? "bg-amber-500" : "bg-sky-500"
              }`}
              style={{ width: `${Math.min(100, Math.max(2, top1.confidence_percent))}%` }}
            />
          </div>
        </div>

        {/* Top 5 Candidates List */}
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
            Top Predictions
          </h4>

          <div className="space-y-2.5">
            {classification.predictions.slice(0, 5).map((item, idx) => (
              <div
                key={`${item.label}-${idx}`}
                className="p-2.5 rounded-xl bg-slate-50/80 border border-slate-100 flex flex-col gap-1.5"
              >
                <div className="flex items-center justify-between text-xs font-medium">
                  <span className="text-slate-800 capitalize flex items-center gap-1.5">
                    <span className="w-4 h-4 rounded-full bg-slate-200 text-slate-600 text-[10px] flex items-center justify-center font-bold">
                      {idx + 1}
                    </span>
                    {item.label}
                  </span>
                  <span className="text-slate-600 font-semibold">
                    {item.confidence_percent.toFixed(2)}%
                  </span>
                </div>

                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-sky-500 h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, Math.max(1, item.confidence_percent))}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer Timing */}
      <div className="mt-5 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span>Inference Time</span>
        <span className="font-semibold text-slate-600">
          {classification.inference_ms.toFixed(1)} ms
        </span>
      </div>
    </div>
  );
};

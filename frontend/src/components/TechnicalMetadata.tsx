"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Cpu, Server, Terminal, Timer } from "lucide-react";
import { RecognitionResponse } from "../types/api";

interface TechnicalMetadataProps {
  response: RecognitionResponse;
}

export const TechnicalMetadata: React.FC<TechnicalMetadataProps> = ({
  response,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-2 text-slate-800 font-bold text-sm">
          <Terminal className="w-4 h-4 text-sky-600" />
          <span>Technical Telemetry &amp; System Metadata</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>{isOpen ? "Collapse" : "Expand"} Details</span>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="px-6 pb-6 pt-2 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs animate-in fade-in">
          {/* Latency card */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-slate-500 font-semibold mb-1">
              <Timer className="w-3.5 h-3.5 text-sky-500" />
              <span>Total Latency</span>
            </div>
            <div className="text-lg font-bold text-slate-900">
              {response.metadata.total_inference_ms.toFixed(1)} ms
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              End-to-end model pipeline
            </div>
          </div>

          {/* Classifier card */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-slate-500 font-semibold mb-1">
              <Cpu className="w-3.5 h-3.5 text-emerald-500" />
              <span>Classifier Backbone</span>
            </div>
            <div className="text-sm font-bold text-slate-900">
              {response.metadata.classifier_model || "MobileNetV2"}
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              ImageNet-1K (1,000 Classes)
            </div>
          </div>

          {/* Detector card */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-slate-500 font-semibold mb-1">
              <Server className="w-3.5 h-3.5 text-purple-500" />
              <span>Object Detector</span>
            </div>
            <div className="text-sm font-bold text-slate-900">
              {response.metadata.detector_model || "SSD-MobileNetV2-COCO"}
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              COCO-2017 (80 Categories)
            </div>
          </div>

          {/* Image specs */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-slate-500 font-semibold mb-1">
              <Terminal className="w-3.5 h-3.5 text-amber-500" />
              <span>Image Normalized</span>
            </div>
            <div className="text-sm font-bold text-slate-900">
              {response.image.width} × {response.image.height} px
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              3-Channel {response.image.format} (Ratio: {response.image.aspect_ratio})
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

"use client";

import React, { useRef, useState, useEffect } from "react";
import { DetectedObject, ImageInfo } from "../types/api";

interface DetectionViewerProps {
  imageSrc: string;
  imageInfo?: ImageInfo;
  detections: DetectedObject[];
  selectedDetectionIndex: number | null;
  onSelectDetection: (index: number | null) => void;
}

const BOX_COLORS = [
  { border: "border-sky-500", bg: "bg-sky-500", text: "text-sky-500", fill: "rgba(14, 165, 233, 0.15)" },
  { border: "border-emerald-500", bg: "bg-emerald-500", text: "text-emerald-500", fill: "rgba(16, 185, 129, 0.15)" },
  { border: "border-purple-500", bg: "bg-purple-500", text: "text-purple-500", fill: "rgba(168, 85, 247, 0.15)" },
  { border: "border-amber-500", bg: "bg-amber-500", text: "text-amber-500", fill: "rgba(245, 158, 11, 0.15)" },
  { border: "border-rose-500", bg: "bg-rose-500", text: "text-rose-500", fill: "rgba(244, 63, 94, 0.15)" },
  { border: "border-cyan-500", bg: "bg-cyan-500", text: "text-cyan-500", fill: "rgba(6, 182, 212, 0.15)" },
  { border: "border-indigo-500", bg: "bg-indigo-500", text: "text-indigo-500", fill: "rgba(99, 102, 241, 0.15)" },
];

export const DetectionViewer: React.FC<DetectionViewerProps> = ({
  imageSrc,
  detections,
  selectedDetectionIndex,
  onSelectDetection,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [renderedBounds, setRenderedBounds] = useState<{
    width: number;
    height: number;
    left: number;
    top: number;
  } | null>(null);

  // Measure rendered image rect within container for pixel-perfect overlays
  const updateBounds = () => {
    if (!imageRef.current || !containerRef.current) return;
    const imgRect = imageRef.current.getBoundingClientRect();
    const containerRect = containerRef.current.getBoundingClientRect();

    setRenderedBounds({
      width: imgRect.width,
      height: imgRect.height,
      left: imgRect.left - containerRect.left,
      top: imgRect.top - containerRect.top,
    });
  };

  useEffect(() => {
    updateBounds();
    window.addEventListener("resize", updateBounds);
    return () => window.removeEventListener("resize", updateBounds);
  }, [imageSrc, detections]);

  return (
    <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-lg flex flex-col">
      {/* Header bar */}
      <div className="px-4 py-2.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
        <span className="font-medium text-slate-200">Interactive Image Viewer</span>
        <span>
          {detections.length > 0
            ? `${detections.length} Object${detections.length > 1 ? "s" : ""} Overlaid`
            : "Classification View"}
        </span>
      </div>

      {/* Image & Bounding Box Canvas */}
      <div
        ref={containerRef}
        className="relative w-full min-h-[320px] max-h-[580px] bg-slate-950/60 flex items-center justify-center p-4 overflow-hidden select-none"
      >
        <div className="relative inline-block max-w-full max-h-[540px]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imageRef}
            src={imageSrc}
            alt="Recognized Scene"
            onLoad={updateBounds}
            className="max-w-full max-h-[540px] w-auto h-auto object-contain rounded-lg block shadow-md"
          />

          {/* Bounding Box Overlays */}
          {renderedBounds &&
            detections.map((detection, idx) => {
              const [ymin, xmin, ymax, xmax] = detection.box.normalized;
              const isSelected = selectedDetectionIndex === idx;
              const color = BOX_COLORS[idx % BOX_COLORS.length];

              // Normalized coordinates relative to rendered image dimensions
              const boxTop = `${(ymin * 100).toFixed(4)}%`;
              const boxLeft = `${(xmin * 100).toFixed(4)}%`;
              const boxWidth = `${((xmax - xmin) * 100).toFixed(4)}%`;
              const boxHeight = `${((ymax - ymin) * 100).toFixed(4)}%`;

              return (
                <div
                  key={`${detection.label}-${idx}`}
                  style={{
                    top: boxTop,
                    left: boxLeft,
                    width: boxWidth,
                    height: boxHeight,
                  }}
                  onMouseEnter={() => onSelectDetection(idx)}
                  onMouseLeave={() => onSelectDetection(null)}
                  onClick={() => onSelectDetection(isSelected ? null : idx)}
                  className={`absolute cursor-pointer transition-all duration-150 border-2 ${
                    color.border
                  } ${
                    isSelected
                      ? "ring-4 ring-white/40 shadow-2xl z-30 scale-[1.01]"
                      : "hover:ring-2 hover:ring-white/20 z-10 opacity-90 hover:opacity-100"
                  }`}
                >
                  {/* Fill tint on hover/select */}
                  <div
                    className="w-full h-full transition-colors"
                    style={{
                      backgroundColor: isSelected
                        ? color.fill.replace("0.15", "0.3")
                        : color.fill,
                    }}
                  />

                  {/* Label badge */}
                  <div
                    className={`absolute -top-7 left-0 px-2 py-0.5 rounded text-[11px] font-bold text-white whitespace-nowrap shadow-md flex items-center gap-1 transition-transform ${
                      color.bg
                    } ${isSelected ? "scale-105" : ""}`}
                  >
                    <span>{detection.label}</span>
                    <span className="opacity-90 font-normal">
                      {Math.round(detection.confidence * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

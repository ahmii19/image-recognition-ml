"use client";

import React from "react";
import { Activity, CheckCircle2, Cpu, RefreshCw, XCircle } from "lucide-react";
import { HealthResponse } from "../types/api";

interface HeaderProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  healthLoading,
  onRefreshHealth,
}) => {
  const isHealthy = health?.status === "ok" && health.models.classifier && health.models.detector;

  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          {/* Title & Subtitle */}
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-sky-500/10 rounded-xl border border-sky-500/20 text-sky-600">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-slate-900">
                  AI Image Recognition
                </h1>
                <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                  v1.0 API
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Multi-class classification &amp; object detection powered by MobileNetV2 &amp; SSD-COCO
              </p>
            </div>
          </div>

          {/* API Health Status Pill */}
          <div className="flex items-center gap-2 self-start sm:self-auto">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors ${
                healthLoading
                  ? "bg-amber-50 border-amber-200 text-amber-700"
                  : isHealthy
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-rose-50 border-rose-200 text-rose-700"
              }`}
            >
              {healthLoading ? (
                <>
                  <Activity className="w-3.5 h-3.5 animate-pulse text-amber-500" />
                  <span>Checking API...</span>
                </>
              ) : isHealthy ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  <span>Serving Layer Connected</span>
                </>
              ) : (
                <>
                  <XCircle className="w-3.5 h-3.5 text-rose-500" />
                  <span>Service Unavailable</span>
                </>
              )}
            </div>

            <button
              onClick={onRefreshHealth}
              disabled={healthLoading}
              title="Refresh API Connection"
              aria-label="Refresh API Connection"
              className="p-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${healthLoading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

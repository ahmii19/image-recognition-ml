"use client";

import React from "react";
import { AlertCircle, RefreshCw, X } from "lucide-react";

interface ErrorAlertProps {
  title: string;
  message: string;
  code?: string;
  requestId?: string | null;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title,
  message,
  code,
  requestId,
  onRetry,
  onDismiss,
}) => {
  return (
    <div className="bg-rose-50 border border-rose-200 rounded-2xl p-5 shadow-sm animate-in fade-in slide-in-from-top-2">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-xl bg-rose-100 text-rose-600 shrink-0 mt-0.5">
            <AlertCircle className="w-5 h-5" />
          </div>

          <div>
            <h4 className="text-sm font-bold text-rose-900">{title}</h4>
            <p className="text-xs text-rose-700 mt-1">{message}</p>

            {/* Diagnostic reference */}
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-rose-500 mt-2 font-mono">
              {code && <span>Code: {code}</span>}
              {requestId && (
                <>
                  <span>•</span>
                  <span>Ref: {requestId}</span>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-semibold text-xs flex items-center gap-1.5 transition-colors shadow-sm"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          )}

          {onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              className="p-1.5 rounded-lg hover:bg-rose-100 text-rose-500 hover:text-rose-700 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

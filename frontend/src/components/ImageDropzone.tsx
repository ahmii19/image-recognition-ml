"use client";

import React, { useRef, useState } from "react";
import { AlertCircle, FileImage, UploadCloud } from "lucide-react";

interface ImageDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"];
const ALLOWED_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
];

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  onFileSelected,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndProcessFile = (file: File) => {
    setValidationError(null);

    // 1. Check size
    if (file.size === 0) {
      setValidationError("The selected file is empty (0 bytes).");
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setValidationError(
        `File size (${(file.size / (1024 * 1024)).toFixed(
          1
        )} MB) exceeds the maximum limit of ${MAX_SIZE_MB}MB.`
      );
      return;
    }

    // 2. Check extension & MIME
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    const isExtensionValid = ALLOWED_EXTENSIONS.includes(extension);
    const isMimeValid = ALLOWED_MIME_TYPES.includes(file.type) || file.type === "";

    if (!isExtensionValid && !isMimeValid) {
      setValidationError(
        `Unsupported format '.${extension}'. Please upload JPG, PNG, WEBP, or BMP.`
      );
      return;
    }

    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative group cursor-pointer border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center transition-all duration-200 ${
          disabled
            ? "opacity-50 cursor-not-allowed bg-slate-50 border-slate-200"
            : isDragging
            ? "border-sky-500 bg-sky-50/60 scale-[1.005] shadow-lg shadow-sky-500/10"
            : "border-slate-300 hover:border-sky-400 hover:bg-slate-50/80 bg-white shadow-sm"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
          onChange={handleInputChange}
          disabled={disabled}
          className="hidden"
          id="image-upload-input"
          aria-label="Upload Image File"
        />

        <div className="flex flex-col items-center justify-center gap-4">
          <div
            className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-200 ${
              isDragging
                ? "bg-sky-500 text-white shadow-md shadow-sky-500/30 scale-110"
                : "bg-sky-50 text-sky-600 group-hover:bg-sky-100 group-hover:scale-105"
            }`}
          >
            <UploadCloud className="w-8 h-8" />
          </div>

          <div>
            <h3 className="text-base font-semibold text-slate-800">
              Drop your image here, or{" "}
              <span className="text-sky-600 underline underline-offset-2">browse</span>
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Supports JPEG, PNG, WEBP, and BMP up to 10MB
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
            {ALLOWED_EXTENSIONS.map((ext) => (
              <span
                key={ext}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-100 text-[11px] font-medium text-slate-600 border border-slate-200 uppercase"
              >
                <FileImage className="w-3 h-3 text-slate-400" />
                {ext}
              </span>
            ))}
          </div>
        </div>
      </div>

      {validationError && (
        <div className="mt-3 flex items-center gap-2 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs animate-in fade-in slide-in-from-top-1">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-500" />
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
};

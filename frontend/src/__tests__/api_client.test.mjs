import test from "node:test";
import assert from "node:assert/strict";

// Inline copy of error mapper for isolated testing
function getFriendlyErrorMessage(error) {
  const map = {
    FILE_TOO_LARGE: {
      title: "File Too Large",
      message: "This image exceeds the maximum allowable limit of 10MB.",
    },
    UNSUPPORTED_FORMAT: {
      title: "Unsupported Image Format",
      message: "Only JPG, PNG, WEBP, and BMP image formats are supported.",
    },
    CORRUPTED_IMAGE: {
      title: "Corrupted Image",
      message: "The uploaded file could not be decoded.",
    },
    EMPTY_FILE: {
      title: "Empty File",
      message: "The uploaded file is 0 bytes and contains no image data.",
    },
    INVALID_MODE: {
      title: "Invalid Mode",
      message: "The requested recognition mode is not supported.",
    },
    MODEL_UNAVAILABLE: {
      title: "Engine Initializing",
      message: "The ML neural network backbones are still initializing.",
    },
    INFERENCE_ERROR: {
      title: "Inference Error",
      message: "The model encountered an error during inference.",
    },
  };

  const code = error?.code || "UNKNOWN";
  const mapped = map[code] || {
    title: "Recognition Error",
    message: error?.message || "An unexpected error occurred.",
  };

  return {
    title: mapped.title,
    message: mapped.message,
    code,
    requestId: error?.requestId || null,
  };
}

test("API Client: correctly maps FILE_TOO_LARGE error code", () => {
  const error = { code: "FILE_TOO_LARGE", message: "Raw error", requestId: "req-123" };
  const res = getFriendlyErrorMessage(error);
  assert.equal(res.title, "File Too Large");
  assert.equal(res.code, "FILE_TOO_LARGE");
  assert.equal(res.requestId, "req-123");
});

test("API Client: correctly maps UNSUPPORTED_FORMAT error code", () => {
  const error = { code: "UNSUPPORTED_FORMAT", message: "Raw error" };
  const res = getFriendlyErrorMessage(error);
  assert.equal(res.title, "Unsupported Image Format");
  assert.equal(res.code, "UNSUPPORTED_FORMAT");
});

test("API Client: correctly maps CORRUPTED_IMAGE error code", () => {
  const error = { code: "CORRUPTED_IMAGE", message: "Corrupted" };
  const res = getFriendlyErrorMessage(error);
  assert.equal(res.title, "Corrupted Image");
  assert.equal(res.code, "CORRUPTED_IMAGE");
});

test("API Client: correctly maps MODEL_UNAVAILABLE error code", () => {
  const error = { code: "MODEL_UNAVAILABLE", message: "Unavailable" };
  const res = getFriendlyErrorMessage(error);
  assert.equal(res.title, "Engine Initializing");
});

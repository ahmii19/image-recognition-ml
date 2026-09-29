/**
 * Centralized API Client for FastAPI Serving Layer.
 */

import {
  ApiErrorResponse,
  ClassificationResult,
  DetectionResult,
  HealthResponse,
  ModelInfoResponse,
  OpenVocabDetectionResponse,
  OpenVocabResponse,
  ReadyResponse,
  RecognitionMode,
  RecognitionResponse,
  RegionRecognitionResponse,
} from "../types/api";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiClientError extends Error {
  code: string;
  requestId?: string | null;
  statusCode?: number;

  constructor(
    message: string,
    code = "UNKNOWN_ERROR",
    requestId?: string | null,
    statusCode?: number
  ) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.requestId = requestId;
    this.statusCode = statusCode;
  }
}

/**
 * Map raw backend error codes into friendly user messages.
 */
export function getFriendlyErrorMessage(error: unknown): {
  title: string;
  message: string;
  code?: string;
  requestId?: string | null;
} {
  if (error instanceof ApiClientError) {
    const map: Record<string, { title: string; message: string }> = {
      FILE_TOO_LARGE: {
        title: "File Too Large",
        message:
          "This image exceeds the maximum allowable limit of 10MB. Please choose a smaller file.",
      },
      UNSUPPORTED_FORMAT: {
        title: "Unsupported Image Format",
        message:
          "Only JPG, PNG, WEBP, and BMP image formats are supported by the engine.",
      },
      CORRUPTED_IMAGE: {
        title: "Corrupted Image",
        message:
          "The uploaded file could not be decoded. Please verify the file is a valid image.",
      },
      EMPTY_FILE: {
        title: "Empty File",
        message: "The uploaded file is 0 bytes and contains no image data.",
      },
      INVALID_MODE: {
        title: "Invalid Mode",
        message:
          "The requested recognition mode is not supported by the serving layer.",
      },
      MODEL_UNAVAILABLE: {
        title: "Engine Initializing",
        message:
          "The ML neural network backbones are still initializing or unavailable.",
      },
      INFERENCE_ERROR: {
        title: "Inference Error",
        message:
          "The model encountered an error during inference. Please try another image.",
      },
      INTERNAL_ERROR: {
        title: "Server Error",
        message:
          "An unexpected server error occurred. Please check the backend service logs.",
      },
      NETWORK_ERROR: {
        title: "Service Unreachable",
        message:
          "Could not connect to the recognition backend. Please verify FastAPI is running.",
      },
      TIMEOUT: {
        title: "Request Timeout",
        message:
          "The recognition request timed out after waiting for model response.",
      },
    };

    const mapped = map[error.code] || {
      title: "Recognition Error",
      message: error.message || "An unexpected error occurred during analysis.",
    };

    return {
      title: mapped.title,
      message: mapped.message,
      code: error.code,
      requestId: error.requestId,
    };
  }

  if (error instanceof Error) {
    if (error.name === "AbortError") {
      return {
        title: "Request Cancelled",
        message: "The analysis request was cancelled.",
        code: "CANCELLED",
      };
    }
    return {
      title: "Connection Error",
      message: error.message || "Failed to communicate with API server.",
      code: "NETWORK_ERROR",
    };
  }

  return {
    title: "Unknown Error",
    message: "An unexpected error occurred.",
    code: "UNKNOWN",
  };
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errJson: ApiErrorResponse | null = null;
    try {
      errJson = (await response.json()) as ApiErrorResponse;
    } catch {
      // Non-JSON response
    }

    if (errJson && errJson.error) {
      throw new ApiClientError(
        errJson.error.message || `Request failed with status ${response.status}`,
        errJson.error.code || `HTTP_${response.status}`,
        errJson.error.request_id,
        response.status
      );
    }

    throw new ApiClientError(
      `Request failed with status ${response.status} (${response.statusText})`,
      `HTTP_${response.status}`,
      response.headers.get("X-Request-ID"),
      response.status
    );
  }

  return (await response.json()) as T;
}

/**
 * Fetch service health status.
 */
export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: "GET",
      signal,
    });
    return await handleResponse<HealthResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      "Unable to connect to ML serving API.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Fetch ML engine readiness.
 */
export async function checkReady(signal?: AbortSignal): Promise<ReadyResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/ready`, {
      method: "GET",
      signal,
    });
    return await handleResponse<ReadyResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      "Unable to verify ML engine readiness.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Fetch model metadata.
 */
export async function getModelsInfo(
  signal?: AbortSignal
): Promise<ModelInfoResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/models`, {
      method: "GET",
      signal,
    });
    return await handleResponse<ModelInfoResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      "Unable to fetch model architecture metadata.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to /api/v1/recognize for unified classification and/or detection.
 */
export async function recognizeImage(
  file: File,
  options?: {
    mode?: RecognitionMode;
    classification_threshold?: number;
    detection_threshold?: number;
    top_k?: number;
    render_visualization?: boolean;
  },
  signal?: AbortSignal
): Promise<RecognitionResponse> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("mode", options?.mode || "all");
  formData.append(
    "classification_threshold",
    String(options?.classification_threshold ?? 0.4)
  );
  formData.append(
    "detection_threshold",
    String(options?.detection_threshold ?? 0.4)
  );
  formData.append("top_k", String(options?.top_k ?? 5));
  formData.append(
    "render_visualization",
    String(options?.render_visualization ?? false)
  );

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/recognize`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<RecognitionResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network connection error while sending image to recognition API.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to standalone /api/v1/classify endpoint.
 */
export async function classifyImage(
  file: File,
  options?: {
    top_k?: number;
    threshold?: number;
  },
  signal?: AbortSignal
): Promise<ClassificationResult> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("top_k", String(options?.top_k ?? 5));
  formData.append("threshold", String(options?.threshold ?? 0.4));

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/classify`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<ClassificationResult>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network error while calling classification endpoint.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to standalone /api/v1/detect endpoint.
 */
export async function detectImage(
  file: File,
  options?: {
    threshold?: number;
    max_detections?: number;
  },
  signal?: AbortSignal
): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("threshold", String(options?.threshold ?? 0.4));
  formData.append("max_detections", String(options?.max_detections ?? 20));

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/detect`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<DetectionResult>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network error while calling detection endpoint.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to /api/v1/open-vocabulary for zero-shot text-prompted recognition.
 */
export async function recognizeOpenVocab(
  file: File,
  options: {
    text_queries: string[] | string;
    top_k?: number;
    similarity_threshold?: number;
    prompt_template?: string;
  },
  signal?: AbortSignal
): Promise<OpenVocabResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const queriesStr = Array.isArray(options.text_queries)
    ? options.text_queries.join(",")
    : options.text_queries;
  formData.append("text_queries", queriesStr);

  if (options.top_k !== undefined) {
    formData.append("top_k", String(options.top_k));
  }
  if (options.similarity_threshold !== undefined && options.similarity_threshold !== null) {
    formData.append("similarity_threshold", String(options.similarity_threshold));
  }
  if (options.prompt_template) {
    formData.append("prompt_template", options.prompt_template);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/open-vocabulary`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<OpenVocabResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network error while calling open-vocabulary recognition endpoint.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to /api/v1/open-vocabulary/detect for OWL-ViT object localization & bounding boxes.
 */
export async function detectOpenVocab(
  file: File,
  options: {
    text_queries: string[] | string;
    top_k?: number;
    score_threshold?: number;
    prompt_template?: string;
  },
  signal?: AbortSignal
): Promise<OpenVocabDetectionResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const queriesStr = Array.isArray(options.text_queries)
    ? options.text_queries.join(",")
    : options.text_queries;
  formData.append("text_queries", queriesStr);

  if (options.top_k !== undefined) {
    formData.append("top_k", String(options.top_k));
  }
  if (options.score_threshold !== undefined && options.score_threshold !== null) {
    formData.append("score_threshold", String(options.score_threshold));
  }
  if (options.prompt_template) {
    formData.append("prompt_template", options.prompt_template);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/open-vocabulary/detect`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<OpenVocabDetectionResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network error while calling open-vocabulary detection endpoint.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Manually trigger model unload on /api/v1/open-vocabulary/unload.
 */
export async function unloadOpenVocabDetector(
  signal?: AbortSignal
): Promise<{ success: boolean; message: string; unloaded: boolean }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/open-vocabulary/unload`, {
      method: "POST",
      signal,
    });
    return await handleResponse<{ success: boolean; message: string; unloaded: boolean }>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      "Failed to communicate with unload endpoint.",
      "NETWORK_ERROR"
    );
  }
}

/**
 * Send image to /api/v1/open-vocabulary/region-recognition for grounded two-stage region recognition.
 */
export async function recognizeRegions(
  file: File,
  options: {
    text_queries: string[] | string;
    top_k?: number;
    detection_threshold?: number;
    similarity_threshold?: number;
    max_regions?: number;
    crop_padding_percent?: number;
    prompt_template?: string;
  },
  signal?: AbortSignal
): Promise<RegionRecognitionResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const queriesStr = Array.isArray(options.text_queries)
    ? options.text_queries.join(",")
    : options.text_queries;
  formData.append("text_queries", queriesStr);

  if (options.top_k !== undefined) {
    formData.append("top_k", String(options.top_k));
  }
  if (options.detection_threshold !== undefined && options.detection_threshold !== null) {
    formData.append("detection_threshold", String(options.detection_threshold));
  }
  if (options.similarity_threshold !== undefined && options.similarity_threshold !== null) {
    formData.append("similarity_threshold", String(options.similarity_threshold));
  }
  if (options.max_regions !== undefined) {
    formData.append("max_regions", String(options.max_regions));
  }
  if (options.crop_padding_percent !== undefined) {
    formData.append("crop_padding_percent", String(options.crop_padding_percent));
  }
  if (options.prompt_template) {
    formData.append("prompt_template", options.prompt_template);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/open-vocabulary/region-recognition`, {
      method: "POST",
      body: formData,
      signal,
    });
    return await handleResponse<RegionRecognitionResponse>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    if ((err as Error)?.name === "AbortError") throw err;
    throw new ApiClientError(
      "Network error while calling region recognition endpoint.",
      "NETWORK_ERROR"
    );
  }
}




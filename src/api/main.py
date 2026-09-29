"""Main FastAPI Application Entry Point with Lifespan Model Management."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_V1_STR,
    API_VERSION,
    CORS_ORIGINS,
    LOG_LEVEL,
)
from src.api.errors import register_error_handlers
from src.api.routes import health, models, open_vocab, open_vocab_detection, recognition
from src.phase3.engine import AdvancedRecognitionEngine
from src.phase6.engine import OpenVocabEngine
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager

# Configure Logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application Lifespan Context Manager.

    Loads the Phase 3 ML Recognition Engine and Phase 6 OpenCLIP Engine on startup
    and binds the active singletons to application state so models are reused
    across all HTTP requests without redundant reloading.

    NOTE: OWL-ViT is STRICTLY LAZY-LOADED on first request to preserve startup speed
    and memory headroom.
    """
    logger.info("Initializing Phase 3 Advanced Recognition Engine (MobileNetV2 + SSD-MobileNetV2)...")
    startup_start = time.perf_counter()

    # Pre-load Phase 3 models into memory
    engine = AdvancedRecognitionEngine(lazy_load=False)
    app.state.engine = engine

    # Pre-load Phase 6 OpenCLIP model into memory
    logger.info("Initializing Phase 6 Open-Vocabulary Engine (OpenCLIP ViT-B/32)...")
    open_vocab_engine = OpenVocabEngine(lazy_load=False)
    app.state.open_vocab_engine = open_vocab_engine

    elapsed_ms = (time.perf_counter() - startup_start) * 1000.0
    logger.info(f"ML Engines initialized and ready in {elapsed_ms:.1f}ms. (OWL-ViT is lazy-loaded).")

    yield

    logger.info("FastAPI Serving Layer shutting down. Releasing engine resources...")
    app.state.engine = None
    app.state.open_vocab_engine = None
    OWLViTLifecycleManager.unload()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. Register Error Handlers
    register_error_handlers(app)

    # 2. Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 3. Request Correlation ID and Telemetry Middleware
    @app.middleware("http")
    async def request_telemetry_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:10]}"
        request.state.request_id = req_id

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        logger.info(
            f"[{req_id}] {request.method} {request.url.path} "
            f"-> Status {response.status_code} ({process_time_ms:.1f}ms)"
        )
        return response

    # 4. Include Versioned API Routes
    app.include_router(health.router, prefix=API_V1_STR)
    app.include_router(models.router, prefix=API_V1_STR)
    app.include_router(recognition.router, prefix=API_V1_STR)
    app.include_router(open_vocab.router, prefix=API_V1_STR)
    app.include_router(open_vocab_detection.router, prefix=API_V1_STR)

    # 5. Root convenience redirect to /docs
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


# Root Application Instance
app = create_app()

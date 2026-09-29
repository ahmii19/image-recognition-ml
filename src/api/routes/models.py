"""Model Metadata API Routes."""

from fastapi import APIRouter
from src.api.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from src.api.schemas import (
    ClassifierModelInfo,
    DetectorModelInfo,
    GeneralUnderstandingModelInfo,
    ModelInfoResponse,
    OpenVocabDetectionModelInfo,
    OpenVocabModelInfo,
    RegionIntelligenceModelInfo,
)
from src.device import get_device_info, resolve_device
from src.phase6.config import (
    DEFAULT_CROP_PADDING_PERCENT,
    DEFAULT_MAX_REGIONS,
    OPENCLIP_DEVICE,
)
from src.phase6.owlvit.config import (
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
    OWL_VIT_MODEL_ID,
)
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager
from src.phase7.config import VLM_MODEL_ID
from src.phase7.lifecycle import VLMLifecycleManager
from src.phase7.prompts import PromptMode

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="Model Architectures and Specifications",
    description="Exposes safe model metadata, supported formats, and recognition capabilities without leaking filesystem paths.",
)
async def get_models_info() -> ModelInfoResponse:
    """Return model architecture metadata and operational parameters."""
    dev_info = get_device_info()
    owl_status = OWLViTLifecycleManager.get_status()
    vlm_status = VLMLifecycleManager.get_status()
    vlm_supported_modes = [m.value for m in PromptMode if m != PromptMode.CUSTOM]

    return ModelInfoResponse(
        classifier=ClassifierModelInfo(
            model="MobileNetV2",
            vocabulary="ImageNet-1K",
            classes=1000,
            input_shape=[224, 224, 3],
        ),
        detector=DetectorModelInfo(
            model="SSD-MobileNetV2-COCO",
            vocabulary="COCO-2017",
            semantic_categories=80,
            input_format="uint8 RGB [0, 255]",
        ),
        open_vocabulary=OpenVocabModelInfo(
            model="OpenCLIP-ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            supports_text_queries=True,
            supports_bounding_boxes=False,
            device=str(resolve_device(OPENCLIP_DEVICE)),
            max_queries=20,
            max_query_length=128,
        ),
        open_vocabulary_detection=OpenVocabDetectionModelInfo(
            model="OWL-ViT base patch32",
            model_id=OWL_VIT_MODEL_ID,
            supports_text_queries=True,
            supports_bounding_boxes=True,
            loaded=owl_status["loaded"],
            device=owl_status["device"],
            lazy_loaded=True,
            max_queries=MAX_TEXT_QUERIES,
            max_query_length=MAX_TEXT_QUERY_LENGTH,
        ),
        region_intelligence=RegionIntelligenceModelInfo(
            available=True,
            detector="OWL-ViT base patch32",
            recognizer="OpenCLIP-ViT-B-32",
            lazy_loaded=True,
            max_regions=DEFAULT_MAX_REGIONS,
            crop_padding_percent=DEFAULT_CROP_PADDING_PERCENT,
        ),
        general_understanding=GeneralUnderstandingModelInfo(
            available=True,
            model_id=VLM_MODEL_ID,
            loaded=vlm_status["loaded"],
            device=vlm_status["device"],
            lazy_loaded=True,
            supported_modes=vlm_supported_modes,
        ),
        device_info=dev_info,
        supported_formats=sorted(list(ALLOWED_IMAGE_EXTENSIONS)),
        supported_modes=[
            "all",
            "classification",
            "detection",
            "open_vocabulary",
            "open_vocabulary_detection",
            "region_recognition",
            "general_understanding",
        ],
        max_upload_size_mb=MAX_UPLOAD_SIZE_MB,
    )



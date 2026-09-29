"""Unit and integration tests for Phase 6A Open-Vocabulary Engine using OpenCLIP ViT-B/32."""

from pathlib import Path
from PIL import Image
import pytest

from src.phase6.config import (
    DEFAULT_PROMPT_TEMPLATE,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_PRETRAINED,
)
from src.phase6.engine import OpenVocabEngine
from src.phase6.open_vocab_classifier import OpenCLIPClassifier


@pytest.fixture(scope="module")
def openclip_classifier():
    """Module-level singleton instance of OpenCLIPClassifier."""
    return OpenCLIPClassifier(device="cpu")


@pytest.fixture(scope="module")
def open_vocab_engine(openclip_classifier):
    """Module-level singleton instance of OpenVocabEngine."""
    return OpenVocabEngine(classifier=openclip_classifier, lazy_load=False)


@pytest.fixture(scope="module")
def sample_rose_image():
    """Load sample rose image or fallback to generated rose-like image."""
    rose_path = Path("data/phase2/dataset_cache/flower_photos/flower_photos/roses/10090824183_d02c613f10_m.jpg")
    if rose_path.exists():
        return Image.open(rose_path).convert("RGB")
    # Fallback synthetic red image
    return Image.new("RGB", (224, 224), color=(220, 30, 40))


def test_openclip_classifier_initialization(openclip_classifier):
    """Verify OpenCLIP initializes on CPU with ViT-B/32 architecture."""
    assert openclip_classifier.model_name == OPENCLIP_MODEL_NAME
    assert openclip_classifier.pretrained == OPENCLIP_PRETRAINED
    assert str(openclip_classifier.device) == "cpu"
    assert openclip_classifier.model is not None
    assert openclip_classifier.preprocess is not None
    assert openclip_classifier.tokenizer is not None


def test_openclip_zero_shot_ranking(openclip_classifier, sample_rose_image):
    """Verify zero-shot classification ranks true category (rose) higher than negative distractors."""
    queries = ["rose", "sports car", "laptop", "airplane"]
    result = openclip_classifier.classify(sample_rose_image, queries=queries, top_k=4)

    assert result["success"] is True
    assert result["top_match"]["query"] == "rose"
    assert result["top_match"]["rank"] == 1
    assert -1.0 <= result["top_match"]["similarity_score"] <= 1.0
    assert len(result["all_ranked"]) == 4
    assert result["total_queries_evaluated"] == 4
    assert result["inference_time_ms"] > 0.0

    # Ensure ranking order is monotonic descending
    scores = [item["similarity_score"] for item in result["all_ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_openclip_empty_queries_rejection(openclip_classifier, sample_rose_image):
    """Verify classifier rejects empty text queries list."""
    with pytest.raises(ValueError, match="At least one non-empty text query"):
        openclip_classifier.classify(sample_rose_image, queries=[])


def test_openclip_top_k_parameter(openclip_classifier, sample_rose_image):
    """Verify top_k parameter restricts the number of returned matches."""
    queries = ["rose", "sunflower", "tulip", "daisy", "dandelion", "laptop", "car"]
    result = openclip_classifier.classify(sample_rose_image, queries=queries, top_k=3)

    assert result["success"] is True
    assert len(result["all_ranked"]) == 3
    assert result["total_queries_evaluated"] == 7


def test_openclip_similarity_threshold(openclip_classifier, sample_rose_image):
    """Verify similarity threshold filters out low-scoring queries."""
    queries = ["rose", "refrigerator", "submarine"]
    # High threshold should only keep the closest match (rose) or empty if too high
    result = openclip_classifier.classify(
        sample_rose_image,
        queries=queries,
        similarity_threshold=0.20,
    )
    assert result["success"] is True
    for item in result["all_ranked"]:
        assert item["similarity_score"] >= 0.20


def test_open_vocab_engine_integration(open_vocab_engine, sample_rose_image):
    """Verify OpenVocabEngine end-to-end recognize method produces complete response schema."""
    queries = ["rose", "sunflower", "automobile", "laptop"]
    res = open_vocab_engine.recognize(
        image_input=sample_rose_image,
        text_queries=queries,
        top_k=4,
    )

    assert res["success"] is True
    assert "open_vocabulary" in res
    assert "metadata" in res
    assert "image" in res
    assert res["open_vocabulary"]["top_match"]["query"] == "rose"
    assert len(res["open_vocabulary"]["results"]) == 4
    assert res["metadata"]["open_vocab_model"] == f"OpenCLIP-{OPENCLIP_MODEL_NAME}"
    assert res["image"]["width"] == sample_rose_image.width
    assert res["image"]["height"] == sample_rose_image.height


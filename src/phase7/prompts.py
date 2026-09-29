"""Structured prompt system for Phase 7 VLM General Image Understanding.

Provides reusable, typed prompt builders that produce consistent JSON-schema-aligned
prompts. All prompts instruct the VLM to return valid JSON with predictable fields,
enabling the structured response parser in engine.py to reliably extract results.

Prompt Categories:
    - General understanding (scene, subject, context)
    - Detailed description (objects, colours, atmosphere)
    - Domain-specific understanding (document, diagram, medical, chart, map)
    - Minimal/fast description (single-field summaries)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PromptMode(str, Enum):
    """Supported prompt modes for VLM general image understanding."""

    GENERAL = "general"
    DETAILED = "detailed"
    DOCUMENT = "document"
    DIAGRAM = "diagram"
    CHART = "chart"
    MAP = "map"
    MEDICAL = "medical"
    BRIEF = "brief"
    CUSTOM = "custom"


@dataclass
class VLMPrompt:
    """Immutable container for a fully formatted VLM prompt request.

    Attributes:
        mode: The prompt category / analysis type.
        user_text: The formatted text instruction sent as the user turn.
        expected_keys: JSON keys the response parser should extract.
        description: Human-readable description of what this prompt analyses.
    """

    mode: PromptMode
    user_text: str
    expected_keys: List[str]
    description: str
    extra_meta: Dict[str, Any] = field(default_factory=dict)


# ── Prompt Templates ──────────────────────────────────────────────────────────

def build_general_prompt() -> VLMPrompt:
    """General-purpose scene understanding prompt.

    Returns a structured JSON with: scene_type, main_subject, setting,
    objects, activities, mood, image_quality, and a brief summary.
    """
    user_text = (
        "Analyse this image and respond with ONLY a valid JSON object using these exact keys:\n"
        '{\n'
        '  "scene_type": "<one of: photograph, illustration, diagram, document, chart, map, screenshot, medical, other>",\n'
        '  "main_subject": "<primary subject or focal point of the image>",\n'
        '  "setting": "<environment or context where the scene takes place>",\n'
        '  "objects": ["<list of notable objects visible>"],\n'
        '  "activities": ["<list of actions or events happening, or empty list>"],\n'
        '  "mood": "<overall mood or atmosphere: e.g. calm, busy, dramatic, clinical>",\n'
        '  "image_quality": "<one of: high, medium, low, unclear>",\n'
        '  "summary": "<one concise sentence describing the full image>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.GENERAL,
        user_text=user_text,
        expected_keys=[
            "scene_type", "main_subject", "setting", "objects",
            "activities", "mood", "image_quality", "summary",
        ],
        description="General scene understanding: subject, setting, objects, mood, summary.",
    )


def build_detailed_prompt() -> VLMPrompt:
    """Detailed visual description prompt for rich scene analysis.

    Returns a JSON with: primary_subject, secondary_subjects, colors,
    textures, spatial_layout, lighting, atmosphere, style, and caption.
    """
    user_text = (
        "Provide a detailed visual analysis of this image as ONLY a valid JSON object:\n"
        '{\n'
        '  "primary_subject": "<main focus of the image>",\n'
        '  "secondary_subjects": ["<other notable elements>"],\n'
        '  "dominant_colors": ["<top 3-5 dominant colours>"],\n'
        '  "textures": ["<notable textures: e.g. smooth, rough, metallic, organic>"],\n'
        '  "spatial_layout": "<description of how elements are arranged in the frame>",\n'
        '  "lighting": "<lighting conditions: e.g. natural daylight, artificial, dim, backlit>",\n'
        '  "atmosphere": "<emotional or sensory feel of the image>",\n'
        '  "style": "<visual style: e.g. photorealistic, cartoon, hand-drawn, abstract>",\n'
        '  "caption": "<a detailed one-to-two sentence caption describing the image>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.DETAILED,
        user_text=user_text,
        expected_keys=[
            "primary_subject", "secondary_subjects", "dominant_colors",
            "textures", "spatial_layout", "lighting", "atmosphere", "style", "caption",
        ],
        description="Detailed visual analysis: colours, textures, lighting, spatial layout, caption.",
    )


def build_document_prompt() -> VLMPrompt:
    """Document/text image understanding prompt.

    Suitable for screenshots, scanned pages, signs, labels, and text-heavy images.
    """
    user_text = (
        "Analyse this image as a document or text-containing image. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "document_type": "<type: letter, invoice, article, sign, label, form, receipt, other>",\n'
        '  "language": "<detected language of the text, or unknown>",\n'
        '  "title_or_heading": "<main title or heading if visible, else null>",\n'
        '  "key_text_content": "<the most important text content visible in the image>",\n'
        '  "layout": "<layout description: single column, two-column, table, list, etc.>",\n'
        '  "has_images": <true or false>,\n'
        '  "summary": "<brief summary of what this document communicates>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.DOCUMENT,
        user_text=user_text,
        expected_keys=[
            "document_type", "language", "title_or_heading",
            "key_text_content", "layout", "has_images", "summary",
        ],
        description="Document/text image analysis: type, language, text content, layout.",
    )


def build_diagram_prompt() -> VLMPrompt:
    """Scientific or technical diagram understanding prompt."""
    user_text = (
        "Analyse this scientific or technical diagram. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "diagram_type": "<type: flowchart, architecture, circuit, biological, anatomical, physics, chemistry, other>",\n'
        '  "subject_domain": "<domain: biology, computer science, engineering, physics, chemistry, other>",\n'
        '  "components": ["<list of key components, nodes, or elements visible>"],\n'
        '  "relationships": ["<key relationships, arrows, or flows described>"],\n'
        '  "labels_visible": <true or false>,\n'
        '  "complexity": "<one of: simple, moderate, complex>",\n'
        '  "explanation": "<plain-language explanation of what this diagram shows>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.DIAGRAM,
        user_text=user_text,
        expected_keys=[
            "diagram_type", "subject_domain", "components",
            "relationships", "labels_visible", "complexity", "explanation",
        ],
        description="Technical/scientific diagram understanding: type, domain, components, explanation.",
    )


def build_chart_prompt() -> VLMPrompt:
    """Data chart and graph understanding prompt."""
    user_text = (
        "Analyse this data chart or graph. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "chart_type": "<type: bar chart, line chart, pie chart, scatter plot, histogram, heatmap, table, other>",\n'
        '  "title": "<chart title if visible, else null>",\n'
        '  "x_axis_label": "<x-axis label if applicable, else null>",\n'
        '  "y_axis_label": "<y-axis label if applicable, else null>",\n'
        '  "data_categories": ["<list of legend categories or data series>"],\n'
        '  "key_insight": "<the most important takeaway from the data>",\n'
        '  "data_range": "<approximate range or scale of values if readable>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.CHART,
        user_text=user_text,
        expected_keys=[
            "chart_type", "title", "x_axis_label", "y_axis_label",
            "data_categories", "key_insight", "data_range",
        ],
        description="Chart/graph analysis: type, axes, categories, key insight.",
    )


def build_map_prompt() -> VLMPrompt:
    """Geographic or spatial map understanding prompt."""
    user_text = (
        "Analyse this map image. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "map_type": "<type: geographic, street, transit, topographic, floor plan, schematic, other>",\n'
        '  "region_or_area": "<the geographic region, city, building or area shown>",\n'
        '  "scale_visible": <true or false>,\n'
        '  "legend_visible": <true or false>,\n'
        '  "notable_features": ["<key landmarks, routes, or areas labeled>"],\n'
        '  "orientation": "<compass orientation if visible: e.g. north-up, or unknown>",\n'
        '  "purpose": "<inferred purpose of the map: navigation, reference, planning, etc.>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.MAP,
        user_text=user_text,
        expected_keys=[
            "map_type", "region_or_area", "scale_visible", "legend_visible",
            "notable_features", "orientation", "purpose",
        ],
        description="Map/spatial image analysis: type, region, features, orientation, purpose.",
    )


def build_medical_prompt() -> VLMPrompt:
    """Medical/anatomical image understanding prompt.

    IMPORTANT: This returns an observation-only description. It is NOT a
    diagnostic tool and must not be used for clinical decision-making.
    """
    user_text = (
        "Analyse this medical or anatomical image for descriptive and educational purposes ONLY. "
        "This is NOT a diagnostic tool. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "image_modality": "<modality: X-ray, MRI, CT, ultrasound, histology, photograph, diagram, other>",\n'
        '  "body_region": "<anatomical region shown, if identifiable>",\n'
        '  "view_type": "<view or plane: e.g. anterior, lateral, axial, coronal, sagittal, or unknown>",\n'
        '  "notable_structures": ["<visible anatomical structures>"],\n'
        '  "image_quality": "<one of: high, medium, low, unclear>",\n'
        '  "educational_description": "<objective descriptive text for educational understanding only>",\n'
        '  "disclaimer": "This analysis is for descriptive and educational purposes only, not for clinical diagnosis."\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.MEDICAL,
        user_text=user_text,
        expected_keys=[
            "image_modality", "body_region", "view_type", "notable_structures",
            "image_quality", "educational_description", "disclaimer",
        ],
        description="Medical/anatomical image description (educational/descriptive only — NOT diagnostic).",
    )


def build_brief_prompt() -> VLMPrompt:
    """Minimal single-sentence image summary prompt — fastest inference path."""
    user_text = (
        "In one sentence, describe what is shown in this image. "
        "Respond with ONLY a valid JSON object:\n"
        '{\n'
        '  "summary": "<one clear, concise sentence describing the image>"\n'
        '}\n'
        "Return only the JSON. No markdown, no explanation."
    )
    return VLMPrompt(
        mode=PromptMode.BRIEF,
        user_text=user_text,
        expected_keys=["summary"],
        description="Single-sentence brief image description — fastest inference path.",
    )


def build_custom_prompt(user_instruction: str, expected_keys: Optional[List[str]] = None) -> VLMPrompt:
    """Construct a fully custom prompt from user-supplied instructions.

    Args:
        user_instruction: The exact instruction text to send as the user turn.
        expected_keys: Optional list of JSON keys to extract from the response.
                       If None, the parser will attempt to extract all keys found.

    Returns:
        VLMPrompt instance with mode=CUSTOM.
    """
    return VLMPrompt(
        mode=PromptMode.CUSTOM,
        user_text=user_instruction,
        expected_keys=expected_keys or [],
        description="Custom user-defined prompt.",
    )


# ── Registry ──────────────────────────────────────────────────────────────────

_PROMPT_BUILDERS = {
    PromptMode.GENERAL: build_general_prompt,
    PromptMode.DETAILED: build_detailed_prompt,
    PromptMode.DOCUMENT: build_document_prompt,
    PromptMode.DIAGRAM: build_diagram_prompt,
    PromptMode.CHART: build_chart_prompt,
    PromptMode.MAP: build_map_prompt,
    PromptMode.MEDICAL: build_medical_prompt,
    PromptMode.BRIEF: build_brief_prompt,
}


def get_prompt(mode: PromptMode) -> VLMPrompt:
    """Retrieve a pre-built VLMPrompt by PromptMode.

    Args:
        mode: The desired analysis mode. Must not be CUSTOM (use build_custom_prompt).

    Returns:
        Fully constructed VLMPrompt instance.

    Raises:
        ValueError: If mode is CUSTOM (custom prompts must be built via build_custom_prompt).
        KeyError: If mode is not registered.
    """
    if mode == PromptMode.CUSTOM:
        raise ValueError(
            "PromptMode.CUSTOM prompts must be constructed via build_custom_prompt(user_instruction)."
        )
    return _PROMPT_BUILDERS[mode]()


def list_prompt_modes() -> List[Dict[str, str]]:
    """Return a registry of all available prompt modes and their descriptions."""
    return [
        {"mode": mode.value, "description": builder().description}
        for mode, builder in _PROMPT_BUILDERS.items()
    ]

"""Structured response parser for Phase 7 VLM output.

Handles the full pipeline of extracting and validating JSON from raw VLM text output,
including recovery strategies for malformed JSON, truncated output, and mixed-format
responses that contain markdown fences or prose mixed with JSON.

Recovery Strategy (applied in order):
    1. Direct json.loads() on cleaned text.
    2. Regex extraction of first '{...}' JSON block from text.
    3. Regex extraction from markdown code fence blocks (```json ... ```).
    4. Partial key extraction if full JSON parse is impossible.
    5. Controlled VLMParseError with raw_text preserved for debugging.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("phase7.parser")


class VLMParseError(Exception):
    """Raised when the VLM response cannot be parsed into structured output after all recovery attempts."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class ParseResult:
    """Result of a structured VLM response parse attempt.

    Attributes:
        success: True if a usable structured result was obtained.
        data: Extracted key-value pairs from the VLM response.
        raw_text: The unmodified text output from the VLM.
        recovered: True if recovery strategies were required (not clean JSON).
        recovery_method: Human-readable name of the strategy that succeeded.
        missing_keys: Expected keys that were not found in the response.
        extra_keys: Keys present in the response that were not expected.
        parse_warnings: Non-fatal issues encountered during parsing.
    """

    success: bool
    data: Dict[str, Any]
    raw_text: str
    recovered: bool = False
    recovery_method: str = "direct"
    missing_keys: List[str] = field(default_factory=list)
    extra_keys: List[str] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text, returning inner content."""
    # Match ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first complete {...} JSON object from arbitrary text using bracket matching."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _try_partial_extraction(text: str, expected_keys: List[str]) -> Optional[Dict[str, Any]]:
    """Attempt to extract individual key-value pairs via regex when full JSON parse fails.

    This is a best-effort fallback for severely malformed output. Only string values
    are extracted; complex nested structures are skipped.
    """
    if not expected_keys:
        return None

    extracted: Dict[str, Any] = {}
    for key in expected_keys:
        # Match: "key": "value" patterns
        pattern = re.compile(
            rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']([^"\']+)["\']',
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            extracted[key] = match.group(1).strip()

    return extracted if extracted else None


def parse_vlm_response(
    raw_text: str,
    expected_keys: Optional[List[str]] = None,
) -> ParseResult:
    """Parse raw VLM text output into a structured result dict.

    Applies a 4-stage recovery pipeline:
        Stage 1: Direct JSON parse after basic whitespace clean.
        Stage 2: Strip markdown fences, then re-parse.
        Stage 3: Bracket-match extract first JSON object, then re-parse.
        Stage 4: Per-key regex extraction for partial recovery.

    Args:
        raw_text: The raw string output from the VLM generation step.
        expected_keys: Optional list of JSON keys expected in the response.
                       Used to check for missing/extra keys and guide partial extraction.

    Returns:
        ParseResult — always returns a result object (never raises).
        If all recovery attempts fail, ParseResult.success is False and
        ParseResult.data contains {"raw_output": raw_text}.
    """
    warnings: List[str] = []
    expected = expected_keys or []
    cleaned = raw_text.strip()

    # ── Stage 1: Direct parse ─────────────────────────────────────────────────
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            missing = [k for k in expected if k not in data]
            extra = [k for k in data if k not in expected] if expected else []
            return ParseResult(
                success=True,
                data=data,
                raw_text=raw_text,
                recovered=False,
                recovery_method="direct",
                missing_keys=missing,
                extra_keys=extra,
                parse_warnings=warnings,
            )
        warnings.append(f"Stage 1: JSON parse succeeded but top-level type is {type(data).__name__}, not dict.")
    except json.JSONDecodeError:
        pass

    # ── Stage 2: Strip markdown fences ───────────────────────────────────────
    stripped = _strip_markdown_fences(cleaned)
    if stripped != cleaned:
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                missing = [k for k in expected if k not in data]
                extra = [k for k in data if k not in expected] if expected else []
                return ParseResult(
                    success=True,
                    data=data,
                    raw_text=raw_text,
                    recovered=True,
                    recovery_method="markdown_fence_strip",
                    missing_keys=missing,
                    extra_keys=extra,
                    parse_warnings=warnings,
                )
        except json.JSONDecodeError:
            warnings.append("Stage 2: Markdown fence strip found content but JSON parse failed.")

    # ── Stage 3: Bracket-match first JSON object ──────────────────────────────
    for source_text in (cleaned, stripped):
        json_fragment = _extract_first_json_object(source_text)
        if json_fragment:
            try:
                data = json.loads(json_fragment)
                if isinstance(data, dict):
                    missing = [k for k in expected if k not in data]
                    extra = [k for k in data if k not in expected] if expected else []
                    return ParseResult(
                        success=True,
                        data=data,
                        raw_text=raw_text,
                        recovered=True,
                        recovery_method="json_object_extraction",
                        missing_keys=missing,
                        extra_keys=extra,
                        parse_warnings=warnings,
                    )
            except json.JSONDecodeError:
                warnings.append("Stage 3: Bracket-matched JSON fragment found but failed to parse.")

    # ── Stage 4: Per-key regex extraction ────────────────────────────────────
    if expected:
        partial = _try_partial_extraction(cleaned, expected)
        if partial:
            missing = [k for k in expected if k not in partial]
            warnings.append(
                f"Stage 4: Partial key extraction succeeded with {len(partial)}/{len(expected)} keys. "
                f"Missing: {missing}"
            )
            return ParseResult(
                success=True,
                data=partial,
                raw_text=raw_text,
                recovered=True,
                recovery_method="partial_key_extraction",
                missing_keys=missing,
                extra_keys=[],
                parse_warnings=warnings,
            )

    # ── All stages exhausted ─────────────────────────────────────────────────
    logger.warning(
        "VLM response parser: all recovery strategies exhausted. "
        "Raw output preserved. raw_text preview: %s",
        raw_text[:200],
    )
    warnings.append("All 4 parse recovery stages failed. Raw output preserved.")
    return ParseResult(
        success=False,
        data={"raw_output": raw_text},
        raw_text=raw_text,
        recovered=False,
        recovery_method="none",
        missing_keys=expected,
        extra_keys=[],
        parse_warnings=warnings,
    )

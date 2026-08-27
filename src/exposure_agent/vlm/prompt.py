from __future__ import annotations

import json
from typing import Any

from exposure_agent.models import (
    ExposureMetadata,
    ImageFeatureBundle,
    PreviousRoundFeedback,
    VLMDecision,
)


def build_initial_exposure_prompt(
    *,
    metadata: ExposureMetadata,
    fixed_features: ImageFeatureBundle,
    feedback: PreviousRoundFeedback | None,
) -> str:
    payload = {
        "stage": "initial_exposure_recommendation",
        "task": (
            "Use the original camera image, fixed scene features, and optional previous-round "
            "feedback to propose absolute ISO and shutter targets. Return only JSON."
        ),
        "image_order": (
            ["original_camera_image", "previous_unsatisfactory_result"]
            if feedback is not None
            else ["original_camera_image"]
        ),
        "rules": _rules(),
        "required_schema": _schema(),
        "initial_metadata": metadata.model_dump(),
        "fixed_scene_features": _feature_payload(fixed_features),
        "previous_round_feedback": (
            feedback.model_dump(mode="json") if feedback is not None else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_experience_integration_prompt(
    *,
    metadata: ExposureMetadata,
    fixed_features: ImageFeatureBundle,
    initial_decision: VLMDecision,
    memory_context: dict | None,
    feedback: PreviousRoundFeedback | None,
) -> str:
    payload = {
        "stage": "experience_conditioned_integration",
        "task": (
            "Combine your first recommendation with the retrieved successful exposure "
            "experience. Produce one semi-final absolute ISO and shutter target for local search."
        ),
        "image_order": (
            ["original_camera_image", "previous_unsatisfactory_result"]
            if feedback is not None
            else ["original_camera_image"]
        ),
        "rules": _rules()
        + [
            "The retrieved experience is evidence, not a mandatory answer.",
            "Resolve conflicts using image quality, similarity, and recorded quality gain.",
        ],
        "required_schema": _schema(),
        "initial_metadata": metadata.model_dump(),
        "fixed_scene_features": _feature_payload(fixed_features),
        "first_vlm_recommendation": initial_decision.model_dump(
            mode="json", by_alias=True
        ),
        "retrieved_memory": memory_context,
        "previous_round_feedback": (
            feedback.model_dump(mode="json") if feedback is not None else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_exposure_prompt(
    *,
    metadata: ExposureMetadata,
    fixed_features: ImageFeatureBundle,
    feedback: PreviousRoundFeedback | None = None,
) -> str:
    """Compatibility alias for the first VLM pass."""
    return build_initial_exposure_prompt(
        metadata=metadata,
        fixed_features=fixed_features,
        feedback=feedback,
    )


def _rules() -> list[str]:
    return [
        "Do not return natural language outside JSON.",
        "Do not wrap JSON in Markdown code fences.",
        "ISO and Shutter are absolute target values, not deltas or multipliers.",
        "Do not output EV; EV is derived from ISO and shutter by the controller.",
        "Use ISO to trade exposure against noise and shutter time to trade exposure against blur.",
        "If no adjustment is needed, return the current ISO and shutter values.",
    ]


def _schema() -> dict[str, Any]:
    return {
        "quality": {
            "brightness": "number 0..1",
            "noise": "number 0..1",
            "motion_blur": "number 0..1",
            "highlight": "number 0..1",
            "shadow": "number 0..1",
            "overall_quality": "number 0..1",
        },
        "action": {
            "ISO": "positive integer absolute target ISO",
            "Shutter": "positive number absolute shutter time in seconds",
        },
        "continue": "boolean",
        "reason": "optional string",
    }


def _feature_payload(features: ImageFeatureBundle) -> dict[str, Any]:
    return {
        "feature_version": features.feature_version,
        "histogram_bins": features.histogram_bins,
        "luminance_histogram": features.luminance_histogram,
        "visual_embedding": features.visual_embedding,
        "objective_quality": features.objective_quality.model_dump(),
    }

from __future__ import annotations

import json

from exposure_agent.models import ExposureMetadata
from exposure_agent.vlm.prompt import (
    build_experience_integration_prompt,
    build_initial_exposure_prompt,
)

from tests.conftest import feature_bundle, vlm_decision


def test_initial_prompt_contains_fixed_features_but_no_memory() -> None:
    metadata = ExposureMetadata(image_id="scene", iso=400, shutter_speed_s=1 / 60)
    features = feature_bundle()

    payload = json.loads(
        build_initial_exposure_prompt(
            metadata=metadata,
            fixed_features=features,
            feedback=None,
        )
    )

    assert payload["stage"] == "initial_exposure_recommendation"
    assert payload["initial_metadata"]["iso"] == 400
    assert len(payload["fixed_scene_features"]["luminance_histogram"]) == 32
    assert "retrieved_memory" not in payload
    assert payload["previous_round_feedback"] is None


def test_integration_prompt_contains_initial_decision_and_rag_item() -> None:
    metadata = ExposureMetadata(image_id="scene", iso=400, shutter_speed_s=1 / 60)
    initial = vlm_decision(metadata, target_iso=800, target_shutter=1 / 60)
    memory = {"best_experience": {"final_action": {"ISO": 600, "Shutter": 0.02}}}

    payload = json.loads(
        build_experience_integration_prompt(
            metadata=metadata,
            fixed_features=feature_bundle(),
            initial_decision=initial,
            memory_context=memory,
            feedback=None,
        )
    )

    assert payload["stage"] == "experience_conditioned_integration"
    assert payload["first_vlm_recommendation"]["action"]["ISO"] == 800
    assert payload["retrieved_memory"] == memory
    assert "EV" not in payload["required_schema"]["action"]


def test_prompts_define_absolute_iso_and_shutter_targets() -> None:
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60)
    prompt = build_initial_exposure_prompt(
        metadata=metadata,
        fixed_features=feature_bundle(),
        feedback=None,
    )

    assert "absolute target" in prompt
    assert '"EV"' not in prompt

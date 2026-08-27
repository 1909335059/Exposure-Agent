from __future__ import annotations

import json

from exposure_agent.models import (
    AgentIteration,
    AgentResult,
    ExposureAction,
    ExposureMetadata,
)
from exposure_agent.train import QwenVLTrainingConfig, normalize_messages_for_qwen_vl
from exposure_agent.training import training_example_to_qwen_json, training_examples_from_result

from tests.conftest import feature_bundle, quality_report, vlm_decision


def test_normalize_messages_for_qwen_vl_resolves_image_paths(tmp_path) -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"image": "outputs/previews/0001_001.png"},
                    {"text": json.dumps({"metadata": {"iso": 100}})},
                ],
            },
            {"role": "assistant", "content": '{"action":{"ISO":100}}'},
        ]
    }

    messages = normalize_messages_for_qwen_vl(row, image_root=tmp_path)

    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][0]["image"] == str(
        tmp_path / "outputs/previews/0001_001.png"
    )
    assert messages[1]["content"][1]["type"] == "text"


def test_training_config_defaults_to_three_epochs() -> None:
    config = QwenVLTrainingConfig(
        train_jsonl="training.jsonl",
        output_dir="checkpoint",
    )

    assert config.num_train_epochs == 3.0
    assert config.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.per_device_train_batch_size == 1
    assert config.per_device_eval_batch_size == 1
    assert config.max_length == 4096


def test_training_exports_initial_and_integration_samples_with_search_target(tmp_path) -> None:
    metadata = ExposureMetadata(image_id="scene", iso=100, shutter_speed_s=1 / 60)
    before = quality_report(0.4, acceptable=False, shadow=0.6)
    after = quality_report(0.8, acceptable=True)
    features = feature_bundle(before)
    initial = vlm_decision(metadata, report=before, target_iso=200, target_shutter=1 / 50)
    integrated = vlm_decision(metadata, report=before, target_iso=300, target_shutter=1 / 40)
    best = ExposureAction(target_iso=400, target_shutter_speed_s=1 / 30)
    final_metadata = ExposureMetadata(iso=400, shutter_speed_s=1 / 30, ev=2)
    result = AgentResult(
        run_id="scene",
        scene_id="scene",
        dataset_split="train",
        original_image_path=tmp_path / "input.png",
        initial_metadata=metadata,
        fixed_features=features,
        iterations=[
            AgentIteration(
                index=1,
                original_image_path=tmp_path / "input.png",
                initial_metadata=metadata,
                fixed_features=features,
                initial_vlm_decision=initial,
                memory_context={"best_experience": {"final_action": {"ISO": 320, "Shutter": 0.03}}},
                integrated_vlm_decision=integrated,
                semi_final_action=integrated.action,
                final_action=best,
                final_metadata=final_metadata,
                output_image_path=tmp_path / "output.png",
                objective_quality_before=before,
                objective_quality_after=after,
                quality_gain_from_original=0.4,
                satisfactory=True,
                optimizer_trace={
                    "enabled": True,
                    "best_gain": 0.4,
                    "best_quality": 0.8,
                    "label_source": "local_search_best",
                },
            )
        ],
        final_image_path=tmp_path / "output.png",
        final_metadata=final_metadata,
        final_quality=after.quality,
        final_objective_quality=after,
        stop_reason="quality_satisfactory",
    )

    examples = training_examples_from_result(result)

    assert [example.stage for example in examples] == ["initial", "integration"]
    assert all(example.target_action == best for example in examples)
    integration_json = json.loads(training_example_to_qwen_json(examples[1]))
    user_text = integration_json["messages"][1]["content"][-1]["text"]
    assert "first_vlm_recommendation" in user_text
    assert "retrieved_memory" in user_text
    assert '"EV"' not in integration_json["messages"][-1]["content"]

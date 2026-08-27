from __future__ import annotations

import json

from exposure_agent.memory import JsonlMemory
from exposure_agent.models import (
    AgentIteration,
    AgentResult,
    ExposureAction,
    ExposureMetadata,
)

from tests.conftest import feature_bundle, quality_report, vlm_decision


def _result(
    tmp_path,
    *,
    before: float,
    after: float,
    scene_id: str = "sample",
    physical_scene_id: str | None = None,
) -> AgentResult:
    metadata = ExposureMetadata(
        image_id=scene_id,
        iso=100,
        shutter_speed_s=1 / 60,
    )
    updated = ExposureMetadata(
        image_id=scene_id,
        iso=200,
        shutter_speed_s=1 / 40,
        ev=4.0,
    )
    before_report = quality_report(
        before,
        acceptable=False,
        brightness=0.2,
        shadow=0.5,
    )
    after_report = quality_report(after, acceptable=after >= 0.65)
    features = feature_bundle(before_report)
    initial = vlm_decision(
        metadata,
        report=before_report,
        target_iso=200,
        target_shutter=1 / 40,
    )
    integrated = initial.model_copy(update={"reason": "integrated"})
    final_action = ExposureAction(target_iso=200, target_shutter_speed_s=1 / 40)
    iteration = AgentIteration(
        index=1,
        original_image_path=tmp_path / "input.png",
        initial_metadata=metadata,
        fixed_features=features,
        initial_vlm_decision=initial,
        integrated_vlm_decision=integrated,
        semi_final_action=integrated.action,
        final_action=final_action,
        final_metadata=updated,
        output_image_path=tmp_path / "output.png",
        objective_quality_before=before_report,
        objective_quality_after=after_report,
        quality_gain_from_original=after - before,
        satisfactory=after_report.acceptable,
        optimizer_trace={
            "enabled": True,
            "label_source": "local_search_best",
            "simulator_version": "exposure_simulator_v2_iso_shutter",
        },
    )
    return AgentResult(
        run_id=f"run-{scene_id}",
        scene_id=scene_id,
        physical_scene_id=physical_scene_id,
        original_image_path=tmp_path / "input.png",
        initial_metadata=metadata,
        fixed_features=features,
        iterations=[iteration],
        final_image_path=tmp_path / "output.png",
        final_metadata=updated,
        final_quality=after_report.quality,
        final_objective_quality=after_report,
        stop_reason="quality_satisfactory" if after_report.acceptable else "max_rounds_best_result",
    )


def test_jsonl_memory_appends_only_positive_best_experience(tmp_path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    memory = JsonlMemory(memory_path, min_quality_gain=0.02)

    memory.write(result=_result(tmp_path, before=0.7, after=0.6))
    memory.write(result=_result(tmp_path, before=0.4, after=0.7))

    rows = [json.loads(line) for line in memory_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["quality_gain"] > 0.02
    assert rows[0]["final_action"] == {"ISO": 200, "Shutter": 0.025}
    assert "luminance_histogram" in rows[0]["fixed_features"]


def test_jsonl_memory_retrieves_by_features_histogram_and_initial_action(tmp_path) -> None:
    memory = JsonlMemory(tmp_path / "memory.jsonl")
    stored = _result(tmp_path, before=0.4, after=0.7, scene_id="stored")
    memory.write(result=stored)
    query_metadata = ExposureMetadata(image_id="query", iso=100, shutter_speed_s=1 / 60)
    query_initial = vlm_decision(
        query_metadata,
        report=stored.fixed_features.objective_quality,
        target_iso=200,
        target_shutter=1 / 40,
    )

    context = memory.retrieve(
        fixed_features=stored.fixed_features,
        metadata=query_metadata,
        initial_decision=query_initial,
        scene_id="query",
    )

    assert context is not None
    assert context["retrieval_type"] == "fixed_features_histogram_initial_action"
    assert context["best_experience"]["scene_id"] == "stored"
    assert set(context["best_experience"]["distance_components"]) == {
        "visual",
        "histogram",
        "quality",
        "exposure",
        "initial_action",
    }


def test_jsonl_memory_excludes_same_scene(tmp_path) -> None:
    memory = JsonlMemory(tmp_path / "memory.jsonl")
    stored = _result(tmp_path, before=0.4, after=0.7, scene_id="same")
    memory.write(result=stored)
    metadata = stored.initial_metadata

    context = memory.retrieve(
        fixed_features=stored.fixed_features,
        metadata=metadata,
        initial_decision=stored.iterations[0].initial_vlm_decision,
        scene_id="same",
    )

    assert context is None


def test_jsonl_memory_excludes_different_instance_of_same_physical_scene(tmp_path) -> None:
    memory = JsonlMemory(tmp_path / "memory.jsonl")
    stored = _result(
        tmp_path,
        before=0.4,
        after=0.7,
        scene_id="0001_001",
        physical_scene_id="001",
    )
    memory.write(result=stored)
    metadata = stored.initial_metadata.model_copy(update={"image_id": "0002_001"})

    context = memory.retrieve(
        fixed_features=stored.fixed_features,
        metadata=metadata,
        initial_decision=stored.iterations[0].initial_vlm_decision,
        scene_id="0002_001",
        physical_scene_id="001",
    )

    assert context is None


def test_jsonl_memory_does_not_write_validation_or_read_only_results(tmp_path) -> None:
    validation_path = tmp_path / "validation_memory.jsonl"
    validation_memory = JsonlMemory(validation_path)
    validation_result = _result(tmp_path, before=0.4, after=0.7).model_copy(
        update={"dataset_split": "validation"}
    )
    validation_memory.write(result=validation_result)
    assert validation_path.read_text() == ""

    read_only_path = tmp_path / "readonly.jsonl"
    read_only_path.write_text("sentinel\n")
    read_only_memory = JsonlMemory(read_only_path, read_only=True)
    read_only_memory.write(result=_result(tmp_path, before=0.4, after=0.7))
    assert read_only_path.read_text() == "sentinel\n"

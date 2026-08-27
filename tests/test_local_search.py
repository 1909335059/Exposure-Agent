from __future__ import annotations

import numpy as np
from PIL import Image

from exposure_agent.agent import Policy
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.models import ExposureAction, ExposureMetadata
from exposure_agent.optimizer import LocalSearchOptimizer


def test_two_stage_search_stays_around_integrated_vlm_target(tmp_path) -> None:
    x = np.linspace(0, 50, 24, dtype=np.uint8)
    gray = np.tile(x, (24, 1))
    image_path = tmp_path / "dark.png"
    Image.fromarray(np.stack([gray, gray, gray], axis=2)).save(image_path)
    evaluator = ImageEvaluator()
    policy = Policy()
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60)
    report = evaluator.evaluate_report(image_path)
    semi_final = ExposureAction(target_iso=200, target_shutter_speed_s=1 / 30)
    optimizer = LocalSearchOptimizer(
        evaluator=evaluator,
        policy=policy,
        candidate_dir=tmp_path / "search",
    )

    selected = optimizer.refine_action(
        action=semi_final,
        image_path=image_path,
        metadata=metadata,
        quality=report.quality,
        objective_report=report,
    )

    trace = optimizer.last_trace
    assert trace is not None
    assert trace["search_type"] == "two_stage_around_integrated_vlm_target"
    assert trace["semi_final_action"] == semi_final.model_dump(by_alias=True)
    assert 9 <= trace["candidate_count"] <= 18
    candidate_scores = [row["overall_quality"] for row in trace["candidates"]]
    assert trace["best_quality"] == max([trace["baseline_quality"], *candidate_scores])
    assert selected.model_dump(by_alias=True) == trace["best_action"]
    assert all("EV" not in row["action"] for row in trace["candidates"])


def test_search_keeps_original_target_when_all_candidates_are_worse(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(image_path)
    evaluator = ImageEvaluator()
    policy = Policy()
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60)
    baseline = evaluator.evaluate_report(image_path)
    optimizer = LocalSearchOptimizer(
        evaluator=evaluator,
        policy=policy,
        candidate_dir=tmp_path / "search",
    )
    monkeypatch.setattr(
        optimizer.evaluator,
        "evaluate_report",
        lambda path: baseline.model_copy(
            update={"quality": baseline.quality.model_copy(update={"overall_quality": 0.0})}
        )
        if path != image_path
        else baseline,
    )

    selected = optimizer.refine_action(
        action=ExposureAction(target_iso=800, target_shutter_speed_s=1 / 10),
        image_path=image_path,
        metadata=metadata,
        quality=baseline.quality,
        objective_report=baseline,
    )

    assert selected == ExposureAction.for_metadata(metadata)

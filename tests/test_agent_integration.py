from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.io import savemat

from exposure_agent.agent import ExposureAgent
from exposure_agent.features import ImageFeatureExtractor
from exposure_agent.memory import JsonlMemory
from exposure_agent.models import ExposureAction, ExposureMetadata, VLMDecision
from exposure_agent.vlm import MockVLMClient

from tests.conftest import quality_report, save_gradient_image


def _mock_backend_env() -> dict[str, str]:
    return {**os.environ, "EXPOSURE_BACKEND": "mock"}


class _TrackingVLM:
    def __init__(self, report) -> None:
        self.report = report
        self.initial_feedback = []
        self.integration_calls = 0
        self.integration_memory = []

    def propose_initial(
        self,
        *,
        original_image_path,
        metadata,
        fixed_features,
        feedback=None,
    ) -> VLMDecision:
        self.initial_feedback.append(feedback)
        round_number = len(self.initial_feedback)
        return VLMDecision(
            quality=(feedback.objective_quality if feedback else self.report).quality,
            action=ExposureAction(
                target_iso=metadata.iso * round_number,
                target_shutter_speed_s=metadata.shutter_speed_s * 1.5,
            ),
            continue_adjustment=True,
            reason="tracking_initial",
        )

    def integrate_experience(
        self,
        *,
        original_image_path,
        metadata,
        fixed_features,
        initial_decision,
        memory_context,
        feedback=None,
    ) -> VLMDecision:
        self.integration_calls += 1
        self.integration_memory.append(memory_context)
        return initial_decision.model_copy(update={"reason": "tracking_integration"})


class _FixedMemory:
    def __init__(self, context) -> None:
        self.context = context
        self.retrieved_initial = []

    def retrieve(self, *, initial_decision, **kwargs):
        self.retrieved_initial.append(initial_decision)
        return self.context

    def write(self, *, result) -> None:
        return None


class _PathEvaluator:
    def __init__(self, original_report, output_report) -> None:
        self.original_report = original_report
        self.output_report = output_report
        self.calls = []

    def evaluate_report(self, path):
        self.calls.append(str(path))
        return self.output_report if "round_" in str(path) else self.original_report

    def evaluate(self, path):
        return self.evaluate_report(path).quality


class _CountingFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.count = 0

    def extract(self, image_path):
        self.count += 1
        return super().extract(image_path)


def test_agent_prediction_executes_both_vlm_passes(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    report = quality_report(acceptable=False)
    vlm = _TrackingVLM(report)
    agent = ExposureAgent(vlm=vlm, artifacts_dir=tmp_path / "artifacts")

    prediction = agent.predict_image(
        image_path=image,
        metadata=ExposureMetadata(iso=400, shutter_speed_s=0.01),
    )

    assert len(vlm.initial_feedback) == 1
    assert vlm.integration_calls == 1
    assert prediction.predicted_target_iso == 400
    assert prediction.predicted_target_shutter_speed_s > 0
    assert prediction.initial_vlm_decision.reason == "tracking_initial"
    assert prediction.integrated_vlm_decision.reason == "tracking_integration"


def test_rag_runs_after_first_vlm_and_is_passed_to_second_vlm(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    report = quality_report(acceptable=False)
    vlm = _TrackingVLM(report)
    context = {
        "best_experience": {
            "scene_id": "history",
            "final_action": {"ISO": 800, "Shutter": 0.02},
        }
    }
    memory = _FixedMemory(context)
    agent = ExposureAgent(
        vlm=vlm,
        memory=memory,
        artifacts_dir=tmp_path / "artifacts",
    )

    agent.predict_image(
        image_path=image,
        metadata=ExposureMetadata(iso=400, shutter_speed_s=0.01),
    )

    assert len(memory.retrieved_initial) == 1
    assert memory.retrieved_initial[0].reason == "tracking_initial"
    assert vlm.integration_memory == [context]


def test_unsatisfactory_result_returns_as_feedback_and_caps_at_three_rounds(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    report = quality_report(overall=0.4, acceptable=False, shadow=0.6)
    evaluator = _PathEvaluator(report, report)
    extractor = _CountingFeatureExtractor(evaluator=evaluator)
    vlm = _TrackingVLM(report)
    agent = ExposureAgent(
        vlm=vlm,
        evaluator=evaluator,
        feature_extractor=extractor,
        artifacts_dir=tmp_path / "artifacts",
    )

    result = agent.run_closed_loop_image(
        image_path=image,
        metadata=ExposureMetadata(iso=100, shutter_speed_s=1 / 60),
        max_iterations=9,
        run_id="feedback-test",
    )

    assert len(result.iterations) == 3
    assert vlm.integration_calls == 3
    assert vlm.initial_feedback[0] is None
    assert vlm.initial_feedback[1].round_index == 1
    assert vlm.initial_feedback[1].result_image_path == result.iterations[0].output_image_path
    assert vlm.initial_feedback[2].round_index == 2
    assert extractor.count == 1
    assert result.stop_reason == "max_rounds_best_result"


def test_satisfactory_output_stops_after_first_round_and_writes_memory(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    before = quality_report(overall=0.4, acceptable=False, shadow=0.6)
    after = quality_report(overall=0.8, acceptable=True)
    evaluator = _PathEvaluator(before, after)
    vlm = _TrackingVLM(before)
    memory_path = tmp_path / "memory.jsonl"
    agent = ExposureAgent(
        vlm=vlm,
        evaluator=evaluator,
        memory=JsonlMemory(memory_path),
        artifacts_dir=tmp_path / "artifacts",
    )

    result = agent.run_closed_loop_image(
        image_path=image,
        metadata=ExposureMetadata(iso=100, shutter_speed_s=1 / 60),
        run_id="successful-loop",
        scene_id="scene-a",
    )

    assert len(result.iterations) == 1
    assert result.stop_reason == "quality_satisfactory"
    row = json.loads(memory_path.read_text().strip())
    assert "fixed_features" in row
    assert row["final_action"]["ISO"] == 100
    assert row["quality_gain"] > 0.02


def test_main_cli_outputs_absolute_target_json(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "run",
            str(image),
            "--iso",
            "400",
            "--shutter",
            "0.01",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_mock_backend_env(),
    )

    payload = json.loads(completed.stdout)
    assert payload["predicted_target_iso"] > 0
    assert payload["predicted_target_shutter_speed_s"] > 0
    assert "predicted_ev_delta" not in payload
    assert payload["initial_vlm_decision"]
    assert payload["integrated_vlm_decision"]


def test_main_cli_writes_new_structured_loop_sections(tmp_path) -> None:
    image = save_gradient_image(tmp_path / "input.png")
    structured = tmp_path / "structured.json"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            "run",
            str(image),
            "--iso",
            "400",
            "--shutter",
            "0.01",
            "--mode",
            "loop",
            "--max_iterations",
            "1",
            "--structured_output",
            str(structured),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_mock_backend_env(),
    )

    report = json.loads(structured.read_text())
    round_one = report["rounds"][0]
    assert "vlm_first_recommendation" in round_one
    assert "rag_retrieval" in round_one
    assert "vlm_experience_integrated_recommendation" in round_one
    assert "local_search" in round_one
    assert "output" in round_one


def test_main_cli_runs_sidd_dataset_from_mat_files(tmp_path) -> None:
    root = tmp_path / "SIDD_Small_Raw_Only"
    scene = root / "Data" / "0001_001_S6_00100_00060_3200_L"
    scene.mkdir(parents=True)
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    savemat(scene / "NOISY_RAW_010.MAT", {"x": raw})
    savemat(scene / "GT_RAW_010.MAT", {"x": raw})
    savemat(
        scene / "METADATA_RAW_010.MAT",
        {"metadata": {"ISO": np.array([100]), "ExposureTime": np.array([1 / 60])}},
    )
    output = tmp_path / "outputs" / "predictions.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--dataset",
            "sidd",
            "--data_root",
            str(root),
            "--output",
            str(output),
            "--max_samples",
            "1",
            "--save_rgb_intermediate",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_mock_backend_env(),
    )

    assert "Wrote 1 predictions" in completed.stdout
    row = json.loads(output.read_text().strip())
    assert row["scene_id"] == "0001_001"
    assert row["camera"] == "S6"
    assert row["predicted_target_shutter_speed_s"] > 0
    assert (tmp_path / "outputs" / "rgb" / "0001_001.png").exists()


def test_main_cli_converts_saved_rgb_to_srgb_png(tmp_path) -> None:
    rgb = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(4, 4, 3)
    rgb_path = tmp_path / "linear_rgb.png"
    output = tmp_path / "srgb.png"
    from exposure_agent.dataset import save_rgb_png

    save_rgb_png(rgb, rgb_path)
    completed = subprocess.run(
        [sys.executable, "main.py", "rgb-to-srgb", str(rgb_path), str(output)],
        check=True,
        capture_output=True,
        text=True,
        env=_mock_backend_env(),
    )

    assert "Wrote sRGB preview" in completed.stdout
    assert output.exists()

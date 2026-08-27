from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from exposure_agent.models import (
    AgentResult,
    ExposureAction,
    ImageFeatureBundle,
    TrainingExample,
    VLMDecision,
)
from exposure_agent.vlm.prompt import (
    build_experience_integration_prompt,
    build_initial_exposure_prompt,
)


class TrainingExampleWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)
        self._seen = {
            str(row.get("example_id"))
            for row in _iter_jsonl(self.output_path)
            if row.get("example_id")
        }

    def write_result(self, result: AgentResult) -> int:
        return self.write_examples(training_examples_from_result(result))

    def write_examples(self, examples: Iterable[TrainingExample]) -> int:
        count = 0
        with self.output_path.open("a", encoding="utf-8") as file:
            for example in examples:
                example_id = training_example_id(example)
                if example_id in self._seen:
                    continue
                file.write(training_example_to_qwen_json(example, example_id=example_id))
                file.write("\n")
                self._seen.add(example_id)
                count += 1
        return count


def training_examples_from_result(result: AgentResult) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for iteration in result.iterations:
        trace = iteration.optimizer_trace or {}
        valid = iteration.satisfactory or iteration.quality_gain_from_original >= 0.02
        if not valid:
            continue
        current_report = (
            iteration.feedback_input.objective_quality
            if iteration.feedback_input is not None
            else result.fixed_features.objective_quality
        )
        common = {
            "image_path": str(result.original_image_path),
            "feedback_image_path": (
                str(iteration.feedback_input.result_image_path)
                if iteration.feedback_input is not None
                else None
            ),
            "metadata": result.initial_metadata,
            "fixed_features": result.fixed_features,
            "target_quality": current_report.quality,
            "target_action": iteration.final_action,
            "target_continue": not current_report.acceptable,
            "source": str(trace.get("label_source", "local_search_best")),
            "scene_id": result.scene_id or result.initial_metadata.image_id,
            "physical_scene_id": result.physical_scene_id,
            "dataset_split": result.dataset_split,
            "quality_gain": iteration.quality_gain_from_original,
            "label_score": iteration.objective_quality_after.quality.overall_quality,
            "simulator_version": str(
                trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
            ),
            "quality_calibration_version": current_report.calibration_version,
            "search_summary": _search_summary(trace),
        }
        examples.append(TrainingExample(stage="initial", **common))
        examples.append(
            TrainingExample(
                stage="integration",
                initial_decision=iteration.initial_vlm_decision,
                memory_context=iteration.memory_context,
                **common,
            )
        )
    return examples


def export_training_from_memory(
    *,
    memory_path: str | Path,
    output_path: str | Path,
) -> int:
    writer = TrainingExampleWriter(output_path)
    examples = [
        example
        for row in _iter_jsonl(Path(memory_path))
        if (example := _training_example_from_memory_row(row)) is not None
    ]
    return writer.write_examples(examples)


def _training_example_from_memory_row(row: dict[str, Any]) -> TrainingExample | None:
    try:
        gain = float(row["quality_gain"])
        if row.get("dataset_split") != "train" or gain < 0.02:
            return None
        features = ImageFeatureBundle.model_validate(row["fixed_features"])
        return TrainingExample(
            stage="initial",
            image_path=str(row["original_image_path"]),
            metadata=row["initial_metadata"],
            fixed_features=features,
            target_quality=features.objective_quality.quality,
            target_action=row["final_action"],
            target_continue=not features.objective_quality.acceptable,
            source=str(row.get("label_source", "memory_local_search_best")),
            scene_id=row.get("scene_id"),
            physical_scene_id=row.get("physical_scene_id"),
            dataset_split="train",
            quality_gain=gain,
            label_score=(row.get("quality_after") or {})
            .get("quality", {})
            .get("overall_quality"),
            simulator_version=str(
                row.get("simulator_version", "exposure_simulator_v2_iso_shutter")
            ),
            quality_calibration_version=features.objective_quality.calibration_version,
        )
    except (KeyError, TypeError, ValueError):
        return None


def training_example_id(example: TrainingExample) -> str:
    identity = {
        "stage": example.stage,
        "scene_id": example.scene_id,
        "physical_scene_id": example.physical_scene_id,
        "cross_fold": example.cross_fold,
        "image_path": example.image_path,
        "feedback_image_path": example.feedback_image_path,
        "metadata": example.metadata.model_dump(),
        "split": example.dataset_split,
        "source": example.source,
        "target_action": example.target_action.model_dump(by_alias=True),
        "simulator_version": example.simulator_version,
        "quality_calibration_version": example.quality_calibration_version,
    }
    return sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]


def training_example_to_qwen_json(
    example: TrainingExample,
    *,
    example_id: str | None = None,
) -> str:
    target = {
        "quality": example.target_quality.model_dump(),
        "action": example.target_action.model_dump(by_alias=True),
        "continue": example.target_continue,
        "reason": example.source,
    }
    if example.stage == "initial":
        prompt_text = build_initial_exposure_prompt(
            metadata=example.metadata,
            fixed_features=example.fixed_features,
            feedback=None,
        )
    else:
        if example.initial_decision is None:
            raise ValueError("Integration training examples require initial_decision")
        prompt_text = build_experience_integration_prompt(
            metadata=example.metadata,
            fixed_features=example.fixed_features,
            initial_decision=example.initial_decision,
            memory_context=example.memory_context,
            feedback=None,
        )
    user_content: list[dict[str, str]] = [{"image": example.image_path}]
    if example.feedback_image_path is not None:
        user_content.append({"image": example.feedback_image_path})
    user_content.append({"text": prompt_text})
    payload = {
        "example_id": example_id or training_example_id(example),
        "scene_id": example.scene_id,
        "physical_scene_id": example.physical_scene_id,
        "cross_fold": example.cross_fold,
        "dataset_split": example.dataset_split,
        "stage": example.stage,
        "label_provenance": {
            "source": example.source,
            "quality_gain": example.quality_gain,
            "label_score": example.label_score,
            "simulator_version": example.simulator_version,
            "quality_calibration_version": example.quality_calibration_version,
            "search_summary": example.search_summary,
        },
        "messages": [
            {
                "role": "system",
                "content": "You are an exposure-control VLM. Return valid JSON only.",
            },
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False),
            },
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _search_summary(trace: dict[str, Any]) -> dict[str, Any] | None:
    if not trace.get("enabled"):
        return None
    return {
        "search_type": trace.get("search_type"),
        "candidate_count": trace.get("candidate_count"),
        "baseline_quality": trace.get("baseline_quality"),
        "best_quality": trace.get("best_quality"),
        "best_gain": trace.get("best_gain"),
        "best_action": trace.get("best_action"),
        "best_candidate": trace.get("best_candidate"),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any

from exposure_agent.models import (
    AgentResult,
    DatasetSplit,
    ExposureAction,
    ExposureExperience,
    ExposureMetadata,
    ImageFeatureBundle,
    VLMDecision,
)


class MemoryInterface(ABC):
    @abstractmethod
    def retrieve(
        self,
        *,
        fixed_features: ImageFeatureBundle,
        metadata: ExposureMetadata,
        initial_decision: VLMDecision,
        scene_id: str | None = None,
        physical_scene_id: str | None = None,
        run_id: str | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def write(self, *, result: AgentResult) -> None:
        raise NotImplementedError


class NoOpMemory(MemoryInterface):
    def retrieve(
        self,
        *,
        fixed_features: ImageFeatureBundle,
        metadata: ExposureMetadata,
        initial_decision: VLMDecision,
        scene_id: str | None = None,
        physical_scene_id: str | None = None,
        run_id: str | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> None:
        return None

    def write(self, *, result: AgentResult) -> None:
        return None


class JsonlMemory(MemoryInterface):
    def __init__(
        self,
        path: str | Path,
        *,
        top_k: int = 1,
        evaluator: object | None = None,
        min_quality_gain: float = 0.02,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.top_k = max(1, top_k)
        self.min_quality_gain = min_quality_gain
        self.read_only = read_only
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def retrieve(
        self,
        *,
        fixed_features: ImageFeatureBundle,
        metadata: ExposureMetadata,
        initial_decision: VLMDecision,
        scene_id: str | None = None,
        physical_scene_id: str | None = None,
        run_id: str | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> dict[str, Any] | None:
        excluded_scene = scene_id or metadata.image_id
        candidates: list[tuple[float, dict[str, float], dict[str, Any]]] = []
        for row in self._iter_rows():
            if row.get("dataset_split") != "train":
                continue
            if excluded_scene is not None and row.get("scene_id") == excluded_scene:
                continue
            if (
                physical_scene_id is not None
                and row.get("physical_scene_id") == physical_scene_id
            ):
                continue
            if run_id is not None and row.get("run_id") == run_id:
                continue
            try:
                gain = float(row.get("quality_gain"))
            except (TypeError, ValueError):
                continue
            if gain < self.min_quality_gain:
                continue
            stored_features = row.get("fixed_features")
            stored_metadata = row.get("initial_metadata")
            stored_initial_action = row.get("initial_vlm_action")
            if not all(
                isinstance(value, dict)
                for value in (stored_features, stored_metadata, stored_initial_action)
            ):
                continue
            try:
                components = _distance_components(
                    query_features=fixed_features,
                    query_metadata=metadata,
                    query_action=initial_decision.action,
                    stored_features=stored_features,
                    stored_metadata=stored_metadata,
                    stored_action=stored_initial_action,
                )
            except (KeyError, TypeError, ValueError):
                continue
            score = (
                0.35 * components["visual"]
                + 0.30 * components["histogram"]
                + 0.20 * components["quality"]
                + 0.10 * components["exposure"]
                + 0.05 * components["initial_action"]
            )
            candidates.append((score, components, row))
        if not candidates:
            return None

        nearest = sorted(candidates, key=lambda item: item[0])[: self.top_k]
        examples = [
            _retrieval_item(score=score, components=components, row=row)
            for score, components, row in nearest
        ]
        return {
            "retrieval_type": "fixed_features_histogram_initial_action",
            "query_scene_id": excluded_scene,
            "query_physical_scene_id": physical_scene_id,
            "query_split": dataset_split,
            "excluded_same_scene": True,
            "excluded_same_physical_scene": physical_scene_id is not None,
            "distance_weights": {
                "visual": 0.35,
                "histogram": 0.30,
                "quality": 0.20,
                "exposure": 0.10,
                "initial_action": 0.05,
            },
            "best_experience": examples[0],
            "examples": examples,
        }

    def write(self, *, result: AgentResult) -> None:
        if self.read_only or result.dataset_split != "train" or not result.iterations:
            return
        eligible = [
            iteration
            for iteration in result.iterations
            if iteration.quality_gain_from_original >= self.min_quality_gain
        ]
        if not eligible:
            return
        best = max(
            eligible,
            key=lambda item: item.objective_quality_after.quality.overall_quality or 0.0,
        )
        trace = best.optimizer_trace or {}
        experience = ExposureExperience(
            run_id=result.run_id,
            scene_id=result.scene_id,
            physical_scene_id=result.physical_scene_id,
            dataset_split=result.dataset_split,
            original_image_path=str(result.original_image_path),
            output_image_path=str(best.output_image_path),
            initial_metadata=result.initial_metadata,
            final_metadata=best.final_metadata,
            fixed_features=result.fixed_features,
            initial_vlm_action=best.initial_vlm_decision.action,
            integrated_vlm_action=best.integrated_vlm_decision.action,
            final_action=best.final_action,
            quality_before=result.fixed_features.objective_quality,
            quality_after=best.objective_quality_after,
            quality_gain=best.quality_gain_from_original,
            successful=best.satisfactory,
            label_source=str(trace.get("label_source", "local_search_best")),
            simulator_version=str(
                trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
            ),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(experience.model_dump_json(by_alias=True))
            file.write("\n")

    def _iter_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


def _distance_components(
    *,
    query_features: ImageFeatureBundle,
    query_metadata: ExposureMetadata,
    query_action: ExposureAction,
    stored_features: dict[str, Any],
    stored_metadata: dict[str, Any],
    stored_action: dict[str, Any],
) -> dict[str, float]:
    stored_quality = stored_features["objective_quality"]["quality"]
    query_quality = query_features.objective_quality.quality
    return {
        "visual": _cosine_distance(
            query_features.visual_embedding,
            [float(value) for value in stored_features["visual_embedding"]],
        ),
        "histogram": _jensen_shannon_distance(
            query_features.luminance_histogram,
            [float(value) for value in stored_features["luminance_histogram"]],
        ),
        "quality": _euclidean(
            [
                query_quality.brightness,
                query_quality.noise,
                query_quality.motion_blur,
                query_quality.highlight,
                query_quality.shadow,
                query_features.objective_quality.dynamic_range,
                query_features.objective_quality.midtone_ratio,
            ],
            [
                float(stored_quality["brightness"]),
                float(stored_quality["noise"]),
                float(stored_quality["motion_blur"]),
                float(stored_quality["highlight"]),
                float(stored_quality["shadow"]),
                float(stored_features["objective_quality"]["dynamic_range"]),
                float(stored_features["objective_quality"]["midtone_ratio"]),
            ],
        )
        / math.sqrt(7.0),
        "exposure": _euclidean(
            _exposure_vector(query_metadata),
            _exposure_vector(ExposureMetadata.model_validate(stored_metadata)),
        )
        / math.sqrt(2.0),
        "initial_action": _euclidean(
            _action_vector(query_action, query_metadata),
            _action_vector(
                ExposureAction.model_validate(stored_action),
                ExposureMetadata.model_validate(stored_metadata),
            ),
        )
        / math.sqrt(2.0),
    }


def _retrieval_item(
    *,
    score: float,
    components: dict[str, float],
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "retrieval_score": score,
        "distance_components": components,
        "scene_id": row.get("scene_id"),
        "physical_scene_id": row.get("physical_scene_id"),
        "cross_fold": row.get("cross_fold"),
        "run_id": row.get("run_id"),
        "initial_metadata": row.get("initial_metadata"),
        "final_metadata": row.get("final_metadata"),
        "initial_vlm_action": row.get("initial_vlm_action"),
        "integrated_vlm_action": row.get("integrated_vlm_action"),
        "final_action": row.get("final_action"),
        "quality_before": row.get("quality_before"),
        "quality_after": row.get("quality_after"),
        "quality_gain": row.get("quality_gain"),
        "successful": row.get("successful"),
        "label_source": row.get("label_source"),
    }


def _exposure_vector(metadata: ExposureMetadata) -> list[float]:
    return [
        math.log2(max(metadata.iso, 1) / 100.0) / 8.0,
        math.log2(max(metadata.shutter_speed_s, 1e-9) / (1 / 60)) / 16.0,
    ]


def _action_vector(action: ExposureAction, reference: ExposureMetadata) -> list[float]:
    return [
        math.log2(action.target_iso / reference.iso) / 8.0,
        math.log2(action.target_shutter_speed_s / reference.shutter_speed_s) / 8.0,
    ]


def _euclidean(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (left_norm * right_norm)))


def _jensen_shannon_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 1.0
    epsilon = 1e-12
    left_sum = max(sum(left), epsilon)
    right_sum = max(sum(right), epsilon)
    p = [max(value / left_sum, epsilon) for value in left]
    q = [max(value / right_sum, epsilon) for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]
    divergence = 0.5 * sum(a * math.log(a / m) for a, m in zip(p, midpoint))
    divergence += 0.5 * sum(b * math.log(b / m) for b, m in zip(q, midpoint))
    return math.sqrt(max(divergence, 0.0))

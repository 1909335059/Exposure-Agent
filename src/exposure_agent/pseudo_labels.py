from __future__ import annotations

from pathlib import Path
from math import exp, log
from typing import Literal

from exposure_agent.agent import Policy
from exposure_agent.dataset import ExposureSample, save_rgb_png
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.features import ImageFeatureExtractor
from exposure_agent.memory import JsonlMemory
from exposure_agent.models import (
    ExposureAction,
    ExposureExperience,
    ExposureMetadata,
    ImageFeatureBundle,
    TrainingExample,
    VLMDecision,
)
from exposure_agent.optimizer import LocalSearchOptimizer


DatasetSplit = Literal["train", "validation", "test"]


class SearchPseudoLabelBuilder:
    """Builds auditable absolute ISO/shutter targets without a trained VLM."""

    def __init__(
        self,
        *,
        optimizer: LocalSearchOptimizer,
        evaluator: ImageEvaluator,
        policy: Policy | None = None,
        preview_dir: str | Path = "outputs/previews",
        min_quality_gain: float = 0.02,
    ) -> None:
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.policy = policy or Policy()
        self.feature_extractor = ImageFeatureExtractor(evaluator=evaluator)
        self.preview_dir = Path(preview_dir)
        self.variant_dir = self.preview_dir.parent / "pseudo_label_inputs"
        self.min_quality_gain = min_quality_gain

    def build(
        self,
        sample: ExposureSample,
        *,
        dataset_split: DatasetSplit,
    ) -> tuple[TrainingExample | None, dict]:
        return self.build_variants(
            sample,
            dataset_split=dataset_split,
            exposure_offsets_ev=(0.0,),
        )[0]

    def build_variants(
        self,
        sample: ExposureSample,
        *,
        dataset_split: DatasetSplit,
        exposure_offsets_ev: tuple[float, ...] = (-1.0, 0.0, 1.0),
    ) -> list[tuple[TrainingExample | None, dict]]:
        base_image_path = (
            Path(sample.image_path)
            if sample.image_path is not None
            else save_rgb_png(sample.image, self.preview_dir / f"{sample.scene_id}.png")
        )
        base_metadata = ExposureMetadata(
            image_id=sample.scene_id,
            iso=sample.iso,
            shutter_speed_s=sample.shutter,
            ev=sample.ev or 0.0,
        )
        variants: list[tuple[TrainingExample | None, dict]] = []
        for offset in exposure_offsets_ev:
            if abs(offset) < 1e-12:
                image_path = base_image_path
                metadata = base_metadata
            else:
                variant_target = ExposureAction(
                    target_iso=base_metadata.iso,
                    target_shutter_speed_s=base_metadata.shutter_speed_s * (2.0**offset),
                )
                metadata = self.policy.apply_action(base_metadata, variant_target)
                image_path = self.variant_dir / sample.scene_id / self._variant_name(offset)
                self.optimizer.simulator.render_next_image(
                    source_image_path=base_image_path,
                    previous_metadata=base_metadata,
                    next_metadata=metadata,
                    output_path=image_path,
                )
            variants.append(
                self._build_from_image(
                    image_path=image_path,
                    metadata=metadata,
                    scene_id=sample.scene_id,
                    physical_scene_id=sample.physical_scene_id,
                    dataset_split=dataset_split,
                    source_exposure_offset_ev=offset,
                )
            )
        return variants

    def _build_from_image(
        self,
        *,
        image_path: Path,
        metadata: ExposureMetadata,
        scene_id: str,
        physical_scene_id: str | None,
        dataset_split: DatasetSplit,
        source_exposure_offset_ev: float,
    ) -> tuple[TrainingExample | None, dict]:
        fixed_features = self.feature_extractor.extract(image_path)
        objective = fixed_features.objective_quality
        rule_action = self.policy.heuristic_action(quality=objective, metadata=metadata)
        if objective.acceptable:
            selected = ExposureAction.for_metadata(metadata)
            trace = {
                "enabled": False,
                "reason": "objective_quality_already_acceptable",
                "best_gain": 0.0,
                "best_quality": objective.quality.overall_quality,
                "candidate_count": 0,
                "label_source": "objective_quality_accepted_current_target",
                "simulator_version": self.optimizer.config.simulator_version,
            }
        else:
            selected = self.optimizer.refine_action(
                action=rule_action,
                image_path=image_path,
                metadata=metadata,
                quality=objective.quality,
                objective_report=objective,
            )
            trace = self.optimizer.last_trace or {}
        gain = float(trace.get("best_gain", 0.0) or 0.0)
        valid = objective.acceptable or gain >= self.min_quality_gain
        example = None
        if valid:
            example = TrainingExample(
                stage="initial",
                image_path=str(image_path),
                metadata=metadata,
                fixed_features=fixed_features,
                target_quality=objective.quality,
                target_action=selected,
                target_continue=not objective.acceptable,
                source=str(trace.get("label_source", "local_search_best")),
                scene_id=scene_id,
                physical_scene_id=physical_scene_id,
                dataset_split=dataset_split,
                quality_gain=gain,
                label_score=float(
                    trace.get("best_quality", objective.quality.overall_quality or 0.0)
                ),
                simulator_version=str(
                    trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
                ),
                quality_calibration_version=objective.calibration_version,
                search_summary={
                    key: trace.get(key)
                    for key in (
                        "search_type",
                        "candidate_count",
                        "baseline_quality",
                        "best_quality",
                        "best_gain",
                        "best_action",
                        "best_candidate",
                    )
                }
                | {"source_exposure_offset_ev": source_exposure_offset_ev},
            )
        audit = {
            "scene_id": scene_id,
            "physical_scene_id": physical_scene_id,
            "dataset_split": dataset_split,
            "source_exposure_offset_ev": source_exposure_offset_ev,
            "image_path": str(image_path),
            "metadata": metadata.model_dump(),
            "fixed_features": fixed_features.model_dump(),
            "rule_action": rule_action.model_dump(by_alias=True),
            "selected_action": selected.model_dump(by_alias=True),
            "valid_training_label": valid,
            "quality_gain": gain,
            "search_trace": trace,
        }
        return example, audit

    @staticmethod
    def _variant_name(offset: float) -> str:
        token = f"{offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "x")
        return f"source_exposure_stops_{token}.png"


class CrossFoldIntegrationBuilder:
    """Creates leakage-free integration labels from held-out train groups."""

    def __init__(
        self,
        *,
        optimizer: LocalSearchOptimizer,
        memory: JsonlMemory,
        policy: Policy | None = None,
        min_quality_gain: float = 0.02,
    ) -> None:
        self.optimizer = optimizer
        self.memory = memory
        self.policy = policy or Policy()
        self.min_quality_gain = min_quality_gain

    def build_from_audit(
        self,
        row: dict,
        *,
        cross_fold: int | None,
    ) -> tuple[TrainingExample | None, dict]:
        metadata = ExposureMetadata.model_validate(row["metadata"])
        fixed_features = ImageFeatureBundle.model_validate(row["fixed_features"])
        physical_scene_id = row.get("physical_scene_id")
        dataset_split = row.get("dataset_split", "train")
        initial_action = ExposureAction.model_validate(row["rule_action"])
        initial_decision = VLMDecision(
            quality=fixed_features.objective_quality.quality,
            action=initial_action,
            continue_adjustment=not fixed_features.objective_quality.acceptable,
            reason="rule_teacher_initial_action_v1",
        )
        context = self.memory.retrieve(
            fixed_features=fixed_features,
            metadata=metadata,
            initial_decision=initial_decision,
            scene_id=row.get("scene_id"),
            physical_scene_id=physical_scene_id,
            dataset_split=dataset_split,
        )
        base_audit = {
            "scene_id": row.get("scene_id"),
            "physical_scene_id": physical_scene_id,
            "dataset_split": row.get("dataset_split"),
            "cross_fold": cross_fold,
            "source_exposure_offset_ev": row.get("source_exposure_offset_ev"),
            "image_path": row.get("image_path"),
            "metadata": metadata.model_dump(),
            "fixed_features": fixed_features.model_dump(),
            "initial_decision": initial_decision.model_dump(by_alias=True),
            "memory_context": context,
        }
        if context is None:
            return None, base_audit | {
                "valid_training_label": False,
                "skip_reason": "no_crossfold_memory_match",
            }

        memory_action = ExposureAction.model_validate(
            context["best_experience"]["final_action"]
        )
        memory_weight = _memory_blend_weight(context["best_experience"])
        semi_final = _blend_actions(
            initial_action,
            memory_action,
            memory_weight=memory_weight,
        )
        selected = self.optimizer.refine_action(
            action=semi_final,
            image_path=Path(str(row["image_path"])),
            metadata=metadata,
            quality=fixed_features.objective_quality.quality,
            objective_report=fixed_features.objective_quality,
        )
        trace = self.optimizer.last_trace or {}
        gain = float(trace.get("best_gain", 0.0) or 0.0)
        valid = fixed_features.objective_quality.acceptable or gain >= self.min_quality_gain
        action_is_effective = abs(selected.exposure_change_stops(metadata)) >= 0.05
        context = context | {
            "cross_fold": cross_fold,
            "teacher_integration": {
                "version": "log_exposure_blend_v1",
                "memory_weight": memory_weight,
                "initial_action": initial_action.model_dump(by_alias=True),
                "retrieved_action": memory_action.model_dump(by_alias=True),
                "semi_final_action": semi_final.model_dump(by_alias=True),
            },
        }
        example = None
        if valid:
            example = TrainingExample(
                stage="integration",
                image_path=str(row["image_path"]),
                metadata=metadata,
                fixed_features=fixed_features,
                initial_decision=initial_decision,
                memory_context=context,
                target_quality=fixed_features.objective_quality.quality,
                target_action=selected,
                target_continue=(
                    not fixed_features.objective_quality.acceptable
                    and action_is_effective
                ),
                source=(
                    "crossfold_rag_local_search_teacher_v1"
                    if dataset_split == "train"
                    else "train_memory_heldout_rag_search_teacher_v1"
                ),
                scene_id=row.get("scene_id"),
                physical_scene_id=physical_scene_id,
                cross_fold=cross_fold,
                dataset_split=dataset_split,
                quality_gain=gain,
                label_score=float(
                    trace.get(
                        "best_quality",
                        fixed_features.objective_quality.quality.overall_quality or 0.0,
                    )
                ),
                simulator_version=str(
                    trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
                ),
                quality_calibration_version=(
                    fixed_features.objective_quality.calibration_version
                ),
                search_summary={
                    key: trace.get(key)
                    for key in (
                        "search_type",
                        "candidate_count",
                        "baseline_quality",
                        "best_quality",
                        "best_gain",
                        "best_action",
                        "best_candidate",
                    )
                }
                | {
                    "source_exposure_offset_ev": row.get(
                        "source_exposure_offset_ev"
                    ),
                    "semi_final_action": semi_final.model_dump(by_alias=True),
                    "memory_weight": memory_weight,
                },
            )
        audit = base_audit | {
            "memory_context": context,
            "memory_action": memory_action.model_dump(by_alias=True),
            "memory_weight": memory_weight,
            "semi_final_action": semi_final.model_dump(by_alias=True),
            "selected_action": selected.model_dump(by_alias=True),
            "valid_training_label": valid,
            "quality_gain": gain,
            "search_trace": trace,
        }
        return example, audit


def experience_from_pseudo_audit(
    row: dict,
    *,
    cross_fold: int | None = None,
    min_quality_gain: float = 0.02,
) -> ExposureExperience | None:
    """Convert one positive initial-search record into an append-only experience."""
    gain = float(row.get("quality_gain", 0.0) or 0.0)
    if row.get("dataset_split") != "train" or gain < min_quality_gain:
        return None
    trace = row.get("search_trace") or {}
    best = trace.get("best_candidate")
    if not isinstance(best, dict):
        return None
    offset = float(row.get("source_exposure_offset_ev", 0.0) or 0.0)
    return ExposureExperience(
        run_id=f"pseudo-{row['scene_id']}-{offset:+.2f}",
        scene_id=row.get("scene_id"),
        physical_scene_id=row.get("physical_scene_id"),
        cross_fold=cross_fold,
        dataset_split="train",
        original_image_path=str(row["image_path"]),
        output_image_path=str(best["output_image_path"]),
        initial_metadata=row["metadata"],
        final_metadata=best["updated_metadata"],
        fixed_features=row["fixed_features"],
        initial_vlm_action=row["rule_action"],
        integrated_vlm_action=row["rule_action"],
        final_action=row["selected_action"],
        quality_before=row["fixed_features"]["objective_quality"],
        quality_after=best["objective_quality"],
        quality_gain=gain,
        successful=bool(best["objective_quality"].get("acceptable", False)),
        label_source="train_search_pseudo_label_noisy_srgb_v1",
        simulator_version=str(
            trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
        ),
    )


def _memory_blend_weight(best_experience: dict) -> float:
    distance = max(float(best_experience.get("retrieval_score", 1.0)), 0.0)
    gain = max(float(best_experience.get("quality_gain", 0.0)), 0.0)
    similarity = exp(-3.0 * distance)
    gain_confidence = min(gain / 0.10, 1.0)
    return max(0.20, min(0.75, 0.20 + 0.55 * similarity * gain_confidence))


def _blend_actions(
    initial_action: ExposureAction,
    memory_action: ExposureAction,
    *,
    memory_weight: float,
) -> ExposureAction:
    weight = max(0.0, min(1.0, memory_weight))
    target_iso = int(
        round(
            exp(
                (1.0 - weight) * log(initial_action.target_iso)
                + weight * log(memory_action.target_iso)
            )
        )
    )
    target_shutter = exp(
        (1.0 - weight) * log(initial_action.target_shutter_speed_s)
        + weight * log(memory_action.target_shutter_speed_s)
    )
    return ExposureAction(
        target_iso=max(target_iso, 1),
        target_shutter_speed_s=max(target_shutter, 1e-8),
    )

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image

from exposure_agent.camera import ExposureSimulator
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.models import (
    ExposureAction,
    ExposureMetadata,
    ImageQuality,
    ObjectiveQualityReport,
)


class OptimizerInterface(ABC):
    @abstractmethod
    def refine_action(
        self,
        *,
        action: ExposureAction,
        image_path: str | Path,
        metadata: ExposureMetadata,
        quality: ImageQuality,
        objective_report: ObjectiveQualityReport | None = None,
    ) -> ExposureAction:
        raise NotImplementedError


class NoOpOptimizer(OptimizerInterface):
    last_trace: dict | None = {"enabled": False, "reason": "optimizer_disabled"}

    def refine_action(
        self,
        *,
        action: ExposureAction,
        image_path: str | Path,
        metadata: ExposureMetadata,
        quality: ImageQuality,
        objective_report: ObjectiveQualityReport | None = None,
    ) -> ExposureAction:
        self.last_trace = {
            "enabled": False,
            "reason": "optimizer_disabled",
            "best_action": action.model_dump(by_alias=True),
        }
        return action


@dataclass(frozen=True)
class LocalSearchConfig:
    coarse_iso_step: int = 200
    coarse_shutter_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
    fine_iso_step: int = 100
    fine_shutter_factors: tuple[float, ...] = (0.8, 1.0, 1.25)
    offsets: tuple[int, ...] = (-1, 0, 1)
    simulator_version: str = "exposure_simulator_v2_iso_shutter"
    search_max_dimension: int = 768
    retain_candidate_images: bool = True


class LocalSearchOptimizer(OptimizerInterface):
    def __init__(
        self,
        *,
        simulator: ExposureSimulator | None = None,
        evaluator: ImageEvaluator | None = None,
        policy: object | None = None,
        config: LocalSearchConfig | None = None,
        candidate_dir: str | Path = "artifacts/local_search",
    ) -> None:
        self.simulator = simulator or ExposureSimulator()
        self.evaluator = evaluator or ImageEvaluator()
        if policy is None:
            from exposure_agent.agent.policy import Policy

            policy = Policy()
        self.policy = policy
        self.config = config or LocalSearchConfig()
        self.candidate_dir = Path(candidate_dir)
        self.last_trace: dict | None = None

    def refine_action(
        self,
        *,
        action: ExposureAction,
        image_path: str | Path,
        metadata: ExposureMetadata,
        quality: ImageQuality,
        objective_report: ObjectiveQualityReport | None = None,
    ) -> ExposureAction:
        search_id = uuid4().hex[:10]
        search_dir = self.candidate_dir / search_id
        search_image_path = self._prepare_search_source(Path(image_path), search_dir)
        baseline_report = (
            objective_report
            if search_image_path == Path(image_path) and objective_report is not None
            else self.evaluator.evaluate_report(search_image_path)
        )
        baseline_quality = baseline_report.quality.overall_quality or 0.0
        baseline_action = ExposureAction.for_metadata(metadata)
        trace: list[dict] = []
        seen: set[tuple[int, float]] = set()

        best_action = baseline_action
        best_quality = baseline_quality
        best_candidate: dict | None = None

        coarse = self._neighbors(
            action,
            iso_step=self.config.coarse_iso_step,
            shutter_factors=self.config.coarse_shutter_factors,
        )
        best_action, best_quality, best_candidate = self._evaluate_candidates(
            stage="coarse",
            candidates=coarse,
            proposal=action,
            image_path=search_image_path,
            metadata=metadata,
            search_dir=search_dir,
            baseline_best=(best_action, best_quality, best_candidate),
            trace=trace,
            seen=seen,
        )

        fine = self._neighbors(
            best_action,
            iso_step=self.config.fine_iso_step,
            shutter_factors=self.config.fine_shutter_factors,
        )
        best_action, best_quality, best_candidate = self._evaluate_candidates(
            stage="fine",
            candidates=fine,
            proposal=action,
            image_path=search_image_path,
            metadata=metadata,
            search_dir=search_dir,
            baseline_best=(best_action, best_quality, best_candidate),
            trace=trace,
            seen=seen,
        )

        if not self.config.retain_candidate_images:
            self._prune_search_artifacts(
                candidates=trace,
                best_candidate=best_candidate,
                search_source_path=search_image_path,
                original_image_path=Path(image_path),
            )

        self.last_trace = {
            "enabled": True,
            "search_type": "two_stage_around_integrated_vlm_target",
            "search_id": search_id,
            "simulator_version": self.config.simulator_version,
            "semi_final_action": action.model_dump(by_alias=True),
            "baseline_action": baseline_action.model_dump(by_alias=True),
            "baseline_quality": baseline_quality,
            "candidate_count": len(trace),
            "best_action": best_action.model_dump(by_alias=True),
            "best_quality": best_quality,
            "best_gain": best_quality - baseline_quality,
            "best_candidate": best_candidate,
            "candidate_dir": str(search_dir),
            "search_source_path": str(
                image_path
                if not self.config.retain_candidate_images
                else search_image_path
            ),
            "search_max_dimension": self.config.search_max_dimension,
            "candidates": trace,
            "label_source": "local_search_best",
        }
        return best_action

    @staticmethod
    def _prune_search_artifacts(
        *,
        candidates: list[dict],
        best_candidate: dict | None,
        search_source_path: Path,
        original_image_path: Path,
    ) -> None:
        best_path = (
            Path(best_candidate["output_image_path"])
            if best_candidate is not None
            else None
        )
        for candidate in candidates:
            output_path = Path(candidate["output_image_path"])
            retained = best_path is not None and output_path == best_path
            candidate["image_retained"] = retained
            if not retained and output_path.exists():
                output_path.unlink()
                candidate["output_image_path"] = None
        if search_source_path != original_image_path and search_source_path.exists():
            search_source_path.unlink()

    def _evaluate_candidates(
        self,
        *,
        stage: str,
        candidates: list[ExposureAction],
        proposal: ExposureAction,
        image_path: Path,
        metadata: ExposureMetadata,
        search_dir: Path,
        baseline_best: tuple[ExposureAction, float, dict | None],
        trace: list[dict],
        seen: set[tuple[int, float]],
    ) -> tuple[ExposureAction, float, dict | None]:
        best_action, best_quality, best_candidate = baseline_best
        best_distance = self._proposal_distance(best_action, proposal)
        for candidate in candidates:
            candidate_metadata = self.policy.apply_action(metadata, candidate)
            normalized = ExposureAction.for_metadata(candidate_metadata)
            key = self._action_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            output = search_dir / f"{stage}_{self._candidate_name(normalized)}"
            self.simulator.render_next_image(
                source_image_path=image_path,
                previous_metadata=metadata,
                next_metadata=candidate_metadata,
                output_path=output,
            )
            report = self.evaluator.evaluate_report(output)
            score = report.quality.overall_quality or 0.0
            distance = self._proposal_distance(normalized, proposal)
            candidate_trace = {
                "stage": stage,
                "action": normalized.model_dump(by_alias=True),
                "updated_metadata": candidate_metadata.model_dump(),
                "output_image_path": str(output),
                "objective_quality": report.model_dump(),
                "overall_quality": score,
                "distance_from_semi_final": distance,
            }
            trace.append(candidate_trace)
            if score > best_quality + 1e-9 or (
                abs(score - best_quality) <= 1e-9 and distance < best_distance
            ):
                best_action = normalized
                best_quality = score
                best_candidate = candidate_trace
                best_distance = distance
        return best_action, best_quality, best_candidate

    def _neighbors(
        self,
        center: ExposureAction,
        *,
        iso_step: int,
        shutter_factors: tuple[float, ...],
    ) -> list[ExposureAction]:
        return [
            ExposureAction(
                target_iso=max(1, center.target_iso + offset * iso_step),
                target_shutter_speed_s=max(
                    1e-8,
                    center.target_shutter_speed_s * shutter_factor,
                ),
            )
            for offset in self.config.offsets
            for shutter_factor in shutter_factors
        ]

    def _prepare_search_source(self, image_path: Path, search_dir: Path) -> Path:
        with Image.open(image_path) as image:
            if max(image.size) <= self.config.search_max_dimension:
                return image_path
            resized = image.convert("RGB")
            resized.thumbnail(
                (self.config.search_max_dimension, self.config.search_max_dimension),
                Image.Resampling.LANCZOS,
            )
            output = search_dir / "search_source.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            resized.save(output)
            return output

    @staticmethod
    def _proposal_distance(action: ExposureAction, proposal: ExposureAction) -> float:
        from math import log2

        iso_distance = abs(log2(action.target_iso / proposal.target_iso))
        shutter_distance = abs(
            log2(action.target_shutter_speed_s / proposal.target_shutter_speed_s)
        )
        return iso_distance + shutter_distance

    @staticmethod
    def _action_key(action: ExposureAction) -> tuple[int, float]:
        return action.target_iso, round(action.target_shutter_speed_s, 10)

    @staticmethod
    def _candidate_name(action: ExposureAction) -> str:
        shutter = f"{action.target_shutter_speed_s:.8f}".replace(".", "x")
        return f"iso_{action.target_iso}_shutter_{shutter}.png"

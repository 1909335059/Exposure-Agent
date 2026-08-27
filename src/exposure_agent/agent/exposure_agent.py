from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from exposure_agent.agent.policy import Policy
from exposure_agent.camera import ExposureSimulator
from exposure_agent.dataset.sidd_reader import ExposureSample, save_rgb_png
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.features import ImageFeatureExtractor
from exposure_agent.memory import MemoryInterface, NoOpMemory
from exposure_agent.models import (
    AgentIteration,
    AgentResult,
    DatasetSplit,
    ExposureMetadata,
    ExposurePrediction,
    PreviousRoundFeedback,
)
from exposure_agent.optimizer import NoOpOptimizer, OptimizerInterface
from exposure_agent.vlm import VLMInterface


class ExposureAgent:
    def __init__(
        self,
        *,
        vlm: VLMInterface,
        memory: MemoryInterface | None = None,
        optimizer: OptimizerInterface | None = None,
        simulator: ExposureSimulator | None = None,
        evaluator: ImageEvaluator | None = None,
        feature_extractor: ImageFeatureExtractor | None = None,
        policy: Policy | None = None,
        artifacts_dir: str | Path = "artifacts",
    ) -> None:
        self.vlm = vlm
        self.memory = memory or NoOpMemory()
        self.optimizer = optimizer or NoOpOptimizer()
        self.simulator = simulator or ExposureSimulator()
        self.evaluator = evaluator or ImageEvaluator()
        self.feature_extractor = feature_extractor or ImageFeatureExtractor(
            evaluator=self.evaluator
        )
        self.policy = policy or Policy()
        self.artifacts_dir = Path(artifacts_dir)

    def predict_sample(
        self,
        sample: ExposureSample,
        *,
        dataset_split: DatasetSplit = "train",
    ) -> ExposurePrediction:
        image_path = self._sample_image_path(sample)
        metadata = ExposureMetadata(
            image_id=sample.scene_id,
            iso=sample.iso,
            shutter_speed_s=sample.shutter,
            ev=sample.ev if sample.ev is not None else 0.0,
        )
        return self.predict_image(
            image_path=image_path,
            metadata=metadata,
            scene_id=sample.scene_id,
            physical_scene_id=sample.physical_scene_id,
            camera=sample.camera,
            brightness_level=sample.brightness_level,
            sample_metadata=sample.metadata,
            dataset_split=dataset_split,
        )

    def predict_image(
        self,
        *,
        image_path: str | Path,
        metadata: ExposureMetadata,
        scene_id: str | None = None,
        physical_scene_id: str | None = None,
        camera: str | None = None,
        brightness_level: str | None = None,
        sample_metadata: dict | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> ExposurePrediction:
        path = Path(image_path)
        fixed_features = self.feature_extractor.extract(path)
        initial = self.vlm.propose_initial(
            original_image_path=path,
            metadata=metadata,
            fixed_features=fixed_features,
        )
        memory_context = self.memory.retrieve(
            fixed_features=fixed_features,
            metadata=metadata,
            initial_decision=initial,
            scene_id=scene_id or metadata.image_id,
            physical_scene_id=physical_scene_id,
            dataset_split=dataset_split,
        )
        integrated = self.vlm.integrate_experience(
            original_image_path=path,
            metadata=metadata,
            fixed_features=fixed_features,
            initial_decision=initial,
            memory_context=memory_context,
        )
        final_action = self.optimizer.refine_action(
            action=integrated.action,
            image_path=path,
            metadata=metadata,
            quality=fixed_features.objective_quality.quality,
            objective_report=fixed_features.objective_quality,
        )
        return ExposurePrediction(
            scene_id=scene_id or metadata.image_id,
            physical_scene_id=physical_scene_id,
            camera=camera,
            iso=metadata.iso,
            shutter=metadata.shutter_speed_s,
            ev=metadata.ev,
            brightness_level=brightness_level,
            predicted_target_iso=final_action.target_iso,
            predicted_target_shutter_speed_s=final_action.target_shutter_speed_s,
            predicted_exposure_change_stops=final_action.exposure_change_stops(metadata),
            continue_adjustment=integrated.continue_adjustment,
            quality_score=integrated.quality.overall_quality,
            reason=integrated.reason,
            image_path=str(path),
            metadata=sample_metadata or {},
            fixed_features=fixed_features,
            initial_vlm_decision=initial,
            memory_context=memory_context,
            integrated_vlm_decision=integrated,
            final_action=final_action,
        )

    def run_closed_loop_sample(
        self,
        sample: ExposureSample,
        *,
        max_iterations: int = 3,
        run_id: str | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> AgentResult:
        image_path = self._sample_image_path(sample)
        metadata = ExposureMetadata(
            image_id=sample.scene_id,
            iso=sample.iso,
            shutter_speed_s=sample.shutter,
            ev=sample.ev if sample.ev is not None else 0.0,
        )
        return self.run_closed_loop_image(
            image_path=image_path,
            metadata=metadata,
            max_iterations=max_iterations,
            run_id=run_id or sample.scene_id,
            scene_id=sample.scene_id,
            physical_scene_id=sample.physical_scene_id,
            dataset_split=dataset_split,
        )

    def run_closed_loop_image(
        self,
        *,
        image_path: str | Path,
        metadata: ExposureMetadata,
        max_iterations: int = 3,
        run_id: str | None = None,
        scene_id: str | None = None,
        physical_scene_id: str | None = None,
        dataset_split: DatasetSplit = "train",
    ) -> AgentResult:
        max_rounds = max(1, min(max_iterations, 3))
        original_path = Path(image_path)
        scene_name = scene_id or metadata.image_id
        run_name = run_id or f"{scene_name or original_path.stem}-{uuid4().hex[:8]}"
        run_dir = self.artifacts_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        fixed_features = self.feature_extractor.extract(original_path)
        original_report = fixed_features.objective_quality
        original_score = original_report.quality.overall_quality or 0.0
        feedback: PreviousRoundFeedback | None = None
        iterations: list[AgentIteration] = []

        best_image_path = original_path
        best_metadata = metadata
        best_report = original_report
        best_score = original_score
        stop_reason = "max_rounds_best_result"

        for index in range(1, max_rounds + 1):
            initial = self.vlm.propose_initial(
                original_image_path=original_path,
                metadata=metadata,
                fixed_features=fixed_features,
                feedback=feedback,
            )
            memory_context = self.memory.retrieve(
                fixed_features=fixed_features,
                metadata=metadata,
                initial_decision=initial,
                scene_id=scene_name,
                physical_scene_id=physical_scene_id,
                run_id=run_name,
                dataset_split=dataset_split,
            )
            integrated = self.vlm.integrate_experience(
                original_image_path=original_path,
                metadata=metadata,
                fixed_features=fixed_features,
                initial_decision=initial,
                memory_context=memory_context,
                feedback=feedback,
            )
            final_action = self.optimizer.refine_action(
                action=integrated.action,
                image_path=original_path,
                metadata=metadata,
                quality=original_report.quality,
                objective_report=original_report,
            )
            optimizer_trace = getattr(self.optimizer, "last_trace", None)
            final_metadata = self.policy.apply_action(metadata, final_action)
            output_path = run_dir / f"round_{index:02d}.png"
            self.simulator.render_next_image(
                source_image_path=original_path,
                previous_metadata=metadata,
                next_metadata=final_metadata,
                output_path=output_path,
            )
            after_report = self.evaluator.evaluate_report(output_path)
            after_score = after_report.quality.overall_quality or 0.0
            gain = after_score - original_score
            satisfactory = self.policy.is_satisfactory(after_report)

            if after_score > best_score:
                best_image_path = output_path
                best_metadata = final_metadata
                best_report = after_report
                best_score = after_score

            if satisfactory:
                iteration_stop = "quality_satisfactory"
                stop_reason = iteration_stop
            elif index >= max_rounds:
                iteration_stop = "max_rounds_best_result"
                stop_reason = iteration_stop
            else:
                iteration_stop = None

            iterations.append(
                AgentIteration(
                    index=index,
                    original_image_path=original_path,
                    initial_metadata=metadata,
                    fixed_features=fixed_features,
                    feedback_input=feedback,
                    initial_vlm_decision=initial,
                    memory_context=memory_context,
                    integrated_vlm_decision=integrated,
                    semi_final_action=integrated.action,
                    final_action=final_action,
                    final_metadata=final_metadata,
                    output_image_path=output_path,
                    objective_quality_before=original_report,
                    objective_quality_after=after_report,
                    quality_gain_from_original=gain,
                    satisfactory=satisfactory,
                    optimizer_trace=optimizer_trace,
                    stop_reason=iteration_stop,
                )
            )
            if satisfactory or index >= max_rounds:
                break
            feedback = PreviousRoundFeedback(
                round_index=index,
                result_image_path=output_path,
                selected_action=final_action,
                result_metadata=final_metadata,
                objective_quality=after_report,
                quality_gain_from_original=gain,
                unmet_quality_criteria=self.policy.unmet_quality_criteria(after_report),
            )

        result = AgentResult(
            run_id=run_name,
            original_image_path=original_path,
            initial_metadata=metadata,
            fixed_features=fixed_features,
            iterations=iterations,
            final_image_path=best_image_path,
            final_metadata=best_metadata,
            final_quality=best_report.quality,
            final_objective_quality=best_report,
            stop_reason=stop_reason,
            scene_id=scene_name,
            physical_scene_id=physical_scene_id,
            dataset_split=dataset_split,
        )
        self.memory.write(result=result)
        return result

    def _sample_image_path(self, sample: ExposureSample) -> Path:
        if sample.image_path is not None:
            return Path(sample.image_path)
        output = self.artifacts_dir / "sidd_previews" / f"{sample.scene_id}.png"
        return save_rgb_png(sample.image, output)

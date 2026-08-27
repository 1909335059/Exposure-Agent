from __future__ import annotations

from math import log2
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DatasetSplit = Literal["train", "validation", "test"]


class ExposureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str | None = None
    iso: int = Field(ge=1)
    shutter_speed_s: float = Field(gt=0)
    # EV is derived metadata and is not an independent control action.
    ev: float = 0.0
    aperture: float | None = Field(default=None, gt=0)


class ImageQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brightness: float = Field(ge=0, le=1)
    noise: float = Field(ge=0, le=1)
    motion_blur: float = Field(ge=0, le=1)
    highlight: float = Field(ge=0, le=1)
    shadow: float = Field(ge=0, le=1)
    overall_quality: float | None = Field(default=None, ge=0, le=1)


class ObjectiveQualityReport(BaseModel):
    """No-reference metrics produced outside the VLM."""

    model_config = ConfigDict(extra="forbid")

    quality: ImageQuality
    dynamic_range: float = Field(ge=0, le=1)
    midtone_ratio: float = Field(ge=0, le=1)
    sharpness_confidence: float = Field(ge=0, le=1)
    exposure_score: float = Field(ge=0, le=1)
    contrast_score: float = Field(ge=0, le=1)
    noise_score: float = Field(ge=0, le=1)
    sharpness_score: float = Field(ge=0, le=1)
    acceptable: bool
    calibration_version: str = "no_reference_v2"


class ImageFeatureBundle(BaseModel):
    """Fixed features extracted once from the original camera image."""

    model_config = ConfigDict(extra="forbid")

    luminance_histogram: list[float]
    histogram_bins: int = Field(default=32, ge=8)
    visual_embedding: list[float]
    objective_quality: ObjectiveQualityReport
    feature_version: str = "fixed_scene_features_v1"


class ExposureAction(BaseModel):
    """Absolute ISO and shutter targets returned by both VLM passes."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    target_iso: int = Field(alias="ISO", ge=1)
    target_shutter_speed_s: float = Field(alias="Shutter", gt=0)

    @classmethod
    def for_metadata(cls, metadata: ExposureMetadata) -> "ExposureAction":
        return cls(
            target_iso=metadata.iso,
            target_shutter_speed_s=metadata.shutter_speed_s,
        )

    def exposure_change_stops(self, reference: ExposureMetadata) -> float:
        return log2(self.target_iso / reference.iso) + log2(
            self.target_shutter_speed_s / reference.shutter_speed_s
        )


class VLMDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    quality: ImageQuality
    action: ExposureAction
    continue_adjustment: bool = Field(alias="continue")
    reason: str | None = None


class PreviousRoundFeedback(BaseModel):
    """Dynamic feedback sent to the first VLM pass of the next round."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    result_image_path: Path
    selected_action: ExposureAction
    result_metadata: ExposureMetadata
    objective_quality: ObjectiveQualityReport
    quality_gain_from_original: float
    unmet_quality_criteria: list[str] = Field(default_factory=list)


class AgentIteration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    original_image_path: Path
    initial_metadata: ExposureMetadata
    fixed_features: ImageFeatureBundle
    feedback_input: PreviousRoundFeedback | None = None
    initial_vlm_decision: VLMDecision
    memory_context: dict | None = None
    integrated_vlm_decision: VLMDecision
    semi_final_action: ExposureAction
    final_action: ExposureAction
    final_metadata: ExposureMetadata
    output_image_path: Path
    objective_quality_before: ObjectiveQualityReport
    objective_quality_after: ObjectiveQualityReport
    quality_gain_from_original: float
    satisfactory: bool
    optimizer_trace: dict | None = None
    stop_reason: str | None = None


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    original_image_path: Path
    initial_metadata: ExposureMetadata
    fixed_features: ImageFeatureBundle
    iterations: list[AgentIteration]
    final_image_path: Path
    final_metadata: ExposureMetadata
    final_quality: ImageQuality
    final_objective_quality: ObjectiveQualityReport
    stop_reason: str
    training_example_path: Path | None = None
    scene_id: str | None = None
    physical_scene_id: str | None = None
    dataset_split: DatasetSplit = "train"


class ExposurePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str | None = None
    physical_scene_id: str | None = None
    camera: str | None = None
    iso: int
    shutter: float
    ev: float | None = None
    brightness_level: str | None = None
    predicted_target_iso: int
    predicted_target_shutter_speed_s: float
    predicted_exposure_change_stops: float
    continue_adjustment: bool
    quality_score: float | None = None
    reason: str | None = None
    image_path: str | None = None
    metadata: dict = Field(default_factory=dict)
    fixed_features: ImageFeatureBundle
    initial_vlm_decision: VLMDecision
    memory_context: dict | None = None
    integrated_vlm_decision: VLMDecision
    final_action: ExposureAction


class ExposureExperience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    scene_id: str | None = None
    physical_scene_id: str | None = None
    cross_fold: int | None = Field(default=None, ge=0)
    dataset_split: DatasetSplit = "train"
    original_image_path: str
    output_image_path: str
    initial_metadata: ExposureMetadata
    final_metadata: ExposureMetadata
    fixed_features: ImageFeatureBundle
    initial_vlm_action: ExposureAction
    integrated_vlm_action: ExposureAction
    final_action: ExposureAction
    quality_before: ObjectiveQualityReport
    quality_after: ObjectiveQualityReport
    quality_gain: float
    successful: bool
    label_source: str = "local_search_best"
    simulator_version: str = "exposure_simulator_v2_iso_shutter"


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["initial", "integration"] = "initial"
    image_path: str
    feedback_image_path: str | None = None
    metadata: ExposureMetadata
    fixed_features: ImageFeatureBundle
    initial_decision: VLMDecision | None = None
    memory_context: dict | None = None
    target_quality: ImageQuality
    target_action: ExposureAction
    target_continue: bool
    source: str = "local_search_best"
    scene_id: str | None = None
    physical_scene_id: str | None = None
    cross_fold: int | None = Field(default=None, ge=0)
    dataset_split: DatasetSplit = "train"
    quality_gain: float | None = None
    label_score: float | None = Field(default=None, ge=0, le=1)
    simulator_version: str = "exposure_simulator_v2_iso_shutter"
    quality_calibration_version: str = "no_reference_v2_default"
    search_summary: dict | None = None

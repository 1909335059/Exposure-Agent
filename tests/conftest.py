from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from exposure_agent.models import (
    ExposureAction,
    ExposureMetadata,
    ImageFeatureBundle,
    ImageQuality,
    ObjectiveQualityReport,
    VLMDecision,
)


def save_solid_image(path: Path, value: int, size: tuple[int, int] = (32, 32)) -> Path:
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def save_gradient_image(path: Path, size: tuple[int, int] = (32, 32)) -> Path:
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    arr = np.tile(x, (size[1], 1))
    rgb = np.stack([arr, arr, arr], axis=2)
    Image.fromarray(rgb).save(path)
    return path


def quality_report(
    overall: float = 0.7,
    *,
    acceptable: bool = True,
    brightness: float = 0.4,
    shadow: float = 0.1,
    highlight: float = 0.0,
) -> ObjectiveQualityReport:
    quality = ImageQuality(
        brightness=brightness,
        noise=0.1,
        motion_blur=0.2,
        highlight=highlight,
        shadow=shadow,
        overall_quality=overall,
    )
    return ObjectiveQualityReport(
        quality=quality,
        dynamic_range=0.5,
        midtone_ratio=0.8 if acceptable else 0.4,
        sharpness_confidence=1.0,
        exposure_score=0.8,
        contrast_score=1.0,
        noise_score=0.9,
        sharpness_score=0.8,
        acceptable=acceptable,
    )


def feature_bundle(report: ObjectiveQualityReport | None = None) -> ImageFeatureBundle:
    return ImageFeatureBundle(
        luminance_histogram=[1 / 32] * 32,
        histogram_bins=32,
        visual_embedding=[0.5] * 192,
        objective_quality=report or quality_report(),
    )


def vlm_decision(
    metadata: ExposureMetadata,
    *,
    report: ObjectiveQualityReport | None = None,
    target_iso: int | None = None,
    target_shutter: float | None = None,
    continue_adjustment: bool = True,
) -> VLMDecision:
    objective = report or quality_report()
    return VLMDecision(
        quality=objective.quality,
        action=ExposureAction(
            target_iso=target_iso or metadata.iso,
            target_shutter_speed_s=target_shutter or metadata.shutter_speed_s,
        ),
        continue_adjustment=continue_adjustment,
    )

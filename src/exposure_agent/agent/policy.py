from __future__ import annotations

from dataclasses import dataclass
from math import isclose, log2

from exposure_agent.camera.exposure import compute_relative_ev
from exposure_agent.models import ExposureAction, ExposureMetadata, ObjectiveQualityReport


@dataclass(frozen=True)
class PolicyConfig:
    min_iso: int = 50
    max_iso: int = 12800
    min_shutter_speed_s: float = 1 / 8000
    max_shutter_speed_s: float = 30.0
    min_iso_change: int = 1
    min_shutter_change_stops: float = 0.02


class Policy:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def apply_action(
        self,
        metadata: ExposureMetadata,
        action: ExposureAction,
    ) -> ExposureMetadata:
        cfg = self.config
        iso = int(self._clamp(round(action.target_iso), cfg.min_iso, cfg.max_iso))
        shutter = self._clamp(
            action.target_shutter_speed_s,
            cfg.min_shutter_speed_s,
            cfg.max_shutter_speed_s,
        )
        return ExposureMetadata(
            image_id=metadata.image_id,
            iso=iso,
            shutter_speed_s=shutter,
            ev=compute_relative_ev(iso=iso, shutter=shutter),
            aperture=metadata.aperture,
        )

    def heuristic_action(
        self,
        *,
        quality: ObjectiveQualityReport,
        metadata: ExposureMetadata,
    ) -> ExposureAction:
        metrics = quality.quality
        iso = metadata.iso
        shutter = metadata.shutter_speed_s
        if metrics.highlight > 0.03:
            shutter *= 0.6
        elif metrics.shadow > 0.30 or quality.midtone_ratio < 0.55:
            if metrics.motion_blur > 0.45:
                iso *= 2
            else:
                shutter *= 1.5
        elif metrics.noise > 0.45:
            iso *= 0.5
            shutter *= 2.0
        elif metrics.motion_blur > 0.55 and quality.sharpness_confidence > 0.5:
            iso *= 2
            shutter *= 0.5
        return ExposureAction(target_iso=max(1, round(iso)), target_shutter_speed_s=shutter)

    def is_unchanged(
        self,
        previous: ExposureMetadata,
        updated: ExposureMetadata,
    ) -> bool:
        return previous.iso == updated.iso and isclose(
            previous.shutter_speed_s,
            updated.shutter_speed_s,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    def is_small_action(
        self,
        action: ExposureAction,
        reference: ExposureMetadata,
    ) -> bool:
        shutter_change = abs(
            log2(action.target_shutter_speed_s / reference.shutter_speed_s)
        )
        return (
            abs(action.target_iso - reference.iso) < self.config.min_iso_change
            and shutter_change < self.config.min_shutter_change_stops
        )

    @staticmethod
    def is_satisfactory(report: ObjectiveQualityReport) -> bool:
        return report.acceptable

    @staticmethod
    def unmet_quality_criteria(report: ObjectiveQualityReport) -> list[str]:
        issues: list[str] = []
        if report.quality.shadow > 0.30:
            issues.append("shadow_ratio_too_high")
        if report.quality.highlight > 0.03:
            issues.append("highlight_ratio_too_high")
        if report.midtone_ratio < 0.55:
            issues.append("midtone_ratio_too_low")
        if report.dynamic_range < 0.20:
            issues.append("dynamic_range_too_low")
        if report.quality.noise > 0.45:
            issues.append("noise_too_high")
        if report.quality.motion_blur > 0.55 and report.sharpness_confidence > 0.5:
            issues.append("motion_blur_too_high")
        if (report.quality.overall_quality or 0.0) < 0.65:
            issues.append("overall_quality_too_low")
        return issues

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

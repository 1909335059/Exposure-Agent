from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, laplace

from exposure_agent.models import ImageQuality, ObjectiveQualityReport


@dataclass(frozen=True)
class QualityEvaluatorConfig:
    max_shadow_ratio: float = 0.30
    max_highlight_ratio: float = 0.03
    min_midtone_ratio: float = 0.55
    min_dynamic_range: float = 0.20
    min_overall_quality: float = 0.65
    shadow_threshold: float = 0.05
    highlight_threshold: float = 0.98
    midtone_low: float = 0.10
    midtone_high: float = 0.90
    noise_floor: float = 0.0005
    noise_reference: float = 0.008
    laplacian_reference: float = 0.10
    analysis_max_dimension: int = 768
    exposure_weight: float = 0.40
    contrast_weight: float = 0.20
    noise_weight: float = 0.20
    sharpness_weight: float = 0.20
    calibration_version: str = "no_reference_v2_default"


class ImageEvaluator:
    def __init__(
        self,
        config: QualityEvaluatorConfig | None = None,
        *,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.config = config or QualityEvaluatorConfig()
        if calibration_path is not None:
            self.config = self._load_calibration(calibration_path, self.config)

    def evaluate(self, image_path: str | Path) -> ImageQuality:
        """Compatibility API returning only the compact quality vector."""
        return self.evaluate_report(image_path).quality

    def evaluate_report(self, image_path: str | Path) -> ObjectiveQualityReport:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(
            (self.config.analysis_max_dimension, self.config.analysis_max_dimension),
            Image.Resampling.LANCZOS,
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        luminance = self._luminance(arr)
        cfg = self.config

        brightness = self._clip01(float(np.mean(luminance)))
        shadow = self._clip01(float(np.mean(luminance < cfg.shadow_threshold)))
        highlight = self._clip01(float(np.mean(luminance > cfg.highlight_threshold)))
        midtone_ratio = self._clip01(
            float(np.mean((luminance >= cfg.midtone_low) & (luminance <= cfg.midtone_high)))
        )
        p05, p95 = np.percentile(luminance, [5.0, 95.0])
        dynamic_range = self._clip01(float(p95 - p05))

        noise_raw = self._noise_residual(image, arr)
        noise = self._clip01(
            (noise_raw - cfg.noise_floor)
            / max(cfg.noise_reference - cfg.noise_floor, 1e-8)
        )

        normalized = np.clip(
            (luminance - float(p05)) / max(float(p95 - p05), 1e-6),
            0.0,
            1.0,
        )
        smoothed = gaussian_filter(normalized, sigma=0.6)
        gradient_y, gradient_x = np.gradient(smoothed)
        gradient = np.hypot(gradient_x, gradient_y)
        edge_threshold = float(np.percentile(gradient, 90.0))
        edge_mask = gradient >= edge_threshold
        laplacian_abs = np.abs(laplace(smoothed))
        edge_laplacian = float(np.mean(laplacian_abs[edge_mask]))
        normalized_sharpness = self._clip01(
            edge_laplacian / max(cfg.laplacian_reference, 1e-8)
        )
        contrast_confidence = self._clip01(
            dynamic_range / max(cfg.min_dynamic_range, 1e-8)
        )
        edge_confidence = self._clip01(edge_threshold / 0.02)
        sharpness_confidence = contrast_confidence * edge_confidence
        raw_motion_blur = 1.0 - normalized_sharpness
        # Low contrast does not contain enough evidence for a confident blur label.
        motion_blur = self._clip01(
            sharpness_confidence * raw_motion_blur
            + (1.0 - sharpness_confidence) * 0.5
        )

        shadow_score = self._clip01(
            1.0 - shadow / max(2.0 * cfg.max_shadow_ratio, 1e-8)
        )
        highlight_score = self._clip01(
            1.0 - highlight / max(2.0 * cfg.max_highlight_ratio, 1e-8)
        )
        midtone_score = self._clip01(
            midtone_ratio / max(cfg.min_midtone_ratio, 1e-8)
        )
        exposure_score = self._clip01(
            0.30 * shadow_score + 0.30 * highlight_score + 0.40 * midtone_score
        )
        contrast_score = self._clip01(
            dynamic_range / max(cfg.min_dynamic_range, 1e-8)
        )
        noise_score = 1.0 - noise
        sharpness_score = 1.0 - motion_blur
        overall = self._clip01(
            cfg.exposure_weight * exposure_score
            + cfg.contrast_weight * contrast_score
            + cfg.noise_weight * noise_score
            + cfg.sharpness_weight * sharpness_score
        )

        quality = ImageQuality(
            brightness=brightness,
            noise=noise,
            motion_blur=motion_blur,
            highlight=highlight,
            shadow=shadow,
            overall_quality=overall,
        )
        acceptable = (
            shadow <= cfg.max_shadow_ratio
            and highlight <= cfg.max_highlight_ratio
            and midtone_ratio >= cfg.min_midtone_ratio
            and dynamic_range >= cfg.min_dynamic_range
            and overall >= cfg.min_overall_quality
        )
        return ObjectiveQualityReport(
            quality=quality,
            dynamic_range=dynamic_range,
            midtone_ratio=midtone_ratio,
            sharpness_confidence=sharpness_confidence,
            exposure_score=exposure_score,
            contrast_score=contrast_score,
            noise_score=noise_score,
            sharpness_score=sharpness_score,
            acceptable=acceptable,
            calibration_version=cfg.calibration_version,
        )

    @classmethod
    def calibrate(
        cls,
        image_paths: Iterable[str | Path],
        output_path: str | Path,
    ) -> dict[str, float | str | int]:
        noise_values: list[float] = []
        laplacian_values: list[float] = []
        count = 0
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            luminance = cls._luminance(arr)
            p05, p95 = np.percentile(luminance, [5.0, 95.0])
            normalized = np.clip(
                (luminance - float(p05)) / max(float(p95 - p05), 1e-6),
                0.0,
                1.0,
            )
            noise_values.append(cls._noise_residual(image, arr))
            smoothed = gaussian_filter(normalized, sigma=0.6)
            gradient_y, gradient_x = np.gradient(smoothed)
            gradient = np.hypot(gradient_x, gradient_y)
            edge_mask = gradient >= float(np.percentile(gradient, 90.0))
            laplacian_values.append(float(np.mean(np.abs(laplace(smoothed))[edge_mask])))
            count += 1
        if count == 0:
            raise ValueError("No calibration images were provided")
        noise_floor = max(float(np.percentile(noise_values, 50.0)), 0.0)
        payload: dict[str, float | str | int] = {
            "noise_floor": noise_floor,
            "noise_reference": noise_floor
            + max(0.005, 4.0 * float(np.percentile(noise_values, 90.0))),
            "laplacian_reference": max(
                float(np.percentile(laplacian_values, 50.0)), 1e-5
            ),
            "calibration_version": "no_reference_v2_sidd_gt_srgb",
            "sample_count": count,
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def _load_calibration(
        path: str | Path,
        config: QualityEvaluatorConfig,
    ) -> QualityEvaluatorConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = set(asdict(config))
        updates = {key: value for key, value in payload.items() if key in allowed}
        return replace(config, **updates)

    @staticmethod
    def _luminance(arr: np.ndarray) -> np.ndarray:
        return (
            0.2126 * arr[:, :, 0]
            + 0.7152 * arr[:, :, 1]
            + 0.0722 * arr[:, :, 2]
        )

    @classmethod
    def _noise_residual(cls, image: Image.Image, arr: np.ndarray) -> float:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        blurred_arr = np.asarray(blurred, dtype=np.float32) / 255.0
        luminance = cls._luminance(arr)
        blurred_luminance = cls._luminance(blurred_arr)
        gradient_y, gradient_x = np.gradient(blurred_luminance)
        gradient = np.hypot(gradient_x, gradient_y)
        flat_mask = gradient <= float(np.percentile(gradient, 50.0))
        residual = np.abs(luminance - blurred_luminance)
        return float(np.mean(residual[flat_mask]))

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

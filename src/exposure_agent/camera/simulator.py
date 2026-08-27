from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from exposure_agent.camera.exposure import exposure_scale
from exposure_agent.models import ExposureMetadata


class ExposureSimulator:
    def __init__(self, *, seed: int = 7) -> None:
        self.seed = seed

    def render_next_image(
        self,
        *,
        source_image_path: str | Path,
        previous_metadata: ExposureMetadata,
        next_metadata: ExposureMetadata,
        output_path: str | Path,
    ) -> Path:
        source = Path(source_image_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(source).convert("RGB")
        srgb = np.asarray(image, dtype=np.float32) / 255.0
        linear = self._srgb_to_linear(srgb)
        scaled = linear * exposure_scale(next_metadata, previous_metadata)
        simulated = self._linear_to_srgb(self._tone_map(scaled))
        simulated = self._apply_iso_noise(
            simulated,
            iso=next_metadata.iso,
            reference_iso=previous_metadata.iso,
        )
        simulated = self._apply_shutter_blur(
            simulated,
            shutter=next_metadata.shutter_speed_s,
            reference_shutter=previous_metadata.shutter_speed_s,
        )
        Image.fromarray(np.uint8(np.clip(np.round(simulated * 255.0), 0, 255))).save(output)
        return output

    @staticmethod
    def _tone_map(arr: np.ndarray) -> np.ndarray:
        # A soft shoulder keeps over-exposure visible without turning every
        # brighter candidate into a flat white image.
        mapped = arr / (1.0 + np.maximum(arr - 1.0, 0.0))
        return np.clip(mapped, 0.0, 1.0)

    @staticmethod
    def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
        low = arr <= 0.04045
        return np.where(low, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def _linear_to_srgb(arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, 0.0, 1.0)
        low = arr <= 0.0031308
        return np.where(low, arr * 12.92, 1.055 * np.power(arr, 1 / 2.4) - 0.055)

    def _apply_iso_noise(
        self,
        arr: np.ndarray,
        *,
        iso: int,
        reference_iso: int,
    ) -> np.ndarray:
        iso_ratio = max(iso, 1) / max(reference_iso, 1)
        if iso_ratio <= 1.02:
            return arr
        rng = np.random.default_rng(self.seed + int(iso))
        sigma = min(0.045, 0.006 * np.log2(iso_ratio + 1.0))
        noisy = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        return np.clip(noisy, 0.0, 1.0)

    @staticmethod
    def _apply_shutter_blur(
        arr: np.ndarray,
        *,
        shutter: float,
        reference_shutter: float,
    ) -> np.ndarray:
        ratio = shutter / max(reference_shutter, 1e-9)
        if ratio <= 1.2:
            return arr
        resolution_scale = max(arr.shape[0], arr.shape[1]) / 768.0
        radius = min(
            2.2 * resolution_scale,
            0.35 * np.log2(ratio) * resolution_scale,
        )
        if radius <= 0.05:
            return arr
        image = Image.fromarray(np.uint8(np.clip(np.round(arr * 255.0), 0, 255)))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0

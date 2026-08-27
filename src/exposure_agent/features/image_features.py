from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.models import ImageFeatureBundle


class ImageFeatureExtractor:
    """Extracts deterministic retrieval and prompt features from the original image."""

    def __init__(
        self,
        *,
        evaluator: ImageEvaluator | None = None,
        histogram_bins: int = 32,
        embedding_size: int = 8,
        max_dimension: int = 768,
    ) -> None:
        if histogram_bins < 8:
            raise ValueError("histogram_bins must be at least 8")
        if embedding_size < 2:
            raise ValueError("embedding_size must be at least 2")
        self.evaluator = evaluator or ImageEvaluator()
        self.histogram_bins = histogram_bins
        self.embedding_size = embedding_size
        self.max_dimension = max_dimension

    def extract(self, image_path: str | Path) -> ImageFeatureBundle:
        path = Path(image_path)
        objective = self.evaluator.evaluate_report(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (self.max_dimension, self.max_dimension),
                Image.Resampling.LANCZOS,
            )
            arr = np.asarray(image, dtype=np.float32) / 255.0

        luminance = (
            0.2126 * arr[:, :, 0]
            + 0.7152 * arr[:, :, 1]
            + 0.0722 * arr[:, :, 2]
        )
        histogram, _ = np.histogram(
            luminance,
            bins=self.histogram_bins,
            range=(0.0, 1.0),
        )
        histogram = histogram.astype(np.float64)
        histogram /= max(float(histogram.sum()), 1.0)

        # Percentile normalization makes this compact visual descriptor less
        # sensitive to exposure while retaining coarse color and scene layout.
        normalized = np.empty_like(arr)
        for channel in range(3):
            low, high = np.percentile(arr[:, :, channel], [5.0, 95.0])
            normalized[:, :, channel] = np.clip(
                (arr[:, :, channel] - float(low)) / max(float(high - low), 1e-6),
                0.0,
                1.0,
            )
        descriptor_image = Image.fromarray(np.uint8(np.round(normalized * 255.0)))
        descriptor_image = descriptor_image.resize(
            (self.embedding_size, self.embedding_size),
            Image.Resampling.BILINEAR,
        )
        descriptor = np.asarray(descriptor_image, dtype=np.float32).reshape(-1) / 255.0

        return ImageFeatureBundle(
            luminance_histogram=[round(float(value), 8) for value in histogram],
            histogram_bins=self.histogram_bins,
            visual_embedding=[round(float(value), 6) for value in descriptor],
            objective_quality=objective,
        )

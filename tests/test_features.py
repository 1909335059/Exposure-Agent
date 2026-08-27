from __future__ import annotations

import math

from PIL import Image

from exposure_agent.camera import ExposureSimulator
from exposure_agent.features import ImageFeatureExtractor
from exposure_agent.models import ExposureMetadata


def test_fixed_feature_extractor_outputs_normalized_histogram_and_descriptor(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 16), color=(64, 128, 192)).save(image)

    features = ImageFeatureExtractor().extract(image)

    assert len(features.luminance_histogram) == 32
    assert math.isclose(sum(features.luminance_histogram), 1.0, abs_tol=1e-6)
    assert len(features.visual_embedding) == 8 * 8 * 3
    assert features.feature_version == "fixed_scene_features_v1"


def test_simulator_ignores_ev_when_iso_and_shutter_are_unchanged(tmp_path) -> None:
    image = tmp_path / "input.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (16, 16), color=(80, 120, 160)).save(image)
    simulator = ExposureSimulator()
    reference = ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=-5)
    target = ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=5)

    simulator.render_next_image(
        source_image_path=image,
        previous_metadata=reference,
        next_metadata=target,
        output_path=output,
    )

    assert list(Image.open(output).getdata()) == list(Image.open(image).getdata())

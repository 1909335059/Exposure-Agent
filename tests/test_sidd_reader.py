from __future__ import annotations

import json
import warnings

import numpy as np

from exposure_agent.dataset import PredictionWriter
from exposure_agent.dataset.sidd_reader import (
    SIDDReader,
    SIDDSRGBReader,
    compact_sidd_metadata,
    compute_relative_ev,
    parse_sidd_scene_name,
    raw_to_linear_rgb,
    raw_to_rgb,
    rgb_to_srgb,
)
from exposure_agent.models import ExposureAction, ExposureMetadata, ExposurePrediction

from tests.conftest import feature_bundle, vlm_decision


def test_parse_sidd_scene_name_and_relative_ev() -> None:
    info = parse_sidd_scene_name("0001_001_S6_00100_00060_3200_L")

    assert info.scene_id == "0001_001"
    assert info.camera == "S6"
    assert info.iso == 100
    assert info.shutter == 1 / 60
    assert info.color_temperature == 3200
    assert info.brightness_level == "L"
    assert compute_relative_ev(iso=100, shutter=1 / 60) > 0
    assert info.physical_scene_id == "001"


def test_raw_to_rgb_returns_float_rgb() -> None:
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)

    rgb = raw_to_rgb(raw)

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.float32
    assert 0.0 <= float(rgb.min()) <= float(rgb.max()) <= 1.0


def test_raw_to_rgb_does_not_divide_normalized_float_raw_by_whitelevel() -> None:
    raw = np.linspace(0.0, 0.16, 64, dtype=np.float32).reshape(8, 8)

    rgb = raw_to_rgb(raw, metadata={"metadata": {"WhiteLevel": 1023, "BlackLevel": 0}})

    assert float(rgb.max()) > 0.3


def test_raw_to_linear_rgb_and_rgb_to_srgb_are_separate_steps() -> None:
    raw = np.linspace(0.0, 0.16, 64, dtype=np.float32).reshape(8, 8)

    linear_rgb = raw_to_linear_rgb(raw)
    srgb = rgb_to_srgb(linear_rgb)

    assert linear_rgb.shape == (8, 8, 3)
    assert srgb.shape == linear_rgb.shape
    assert 0.0 <= float(linear_rgb.min()) <= float(linear_rgb.max()) <= 1.0
    assert 0.0 <= float(srgb.min()) <= float(srgb.max()) <= 1.0
    assert float(srgb.mean()) > float(linear_rgb.mean())


def test_sidd_reader_reads_sample_and_saves_preview(tmp_path, monkeypatch) -> None:
    scene = tmp_path / "SIDD_Small_Raw_Only" / "Data" / "0001_001_S6_00100_00060_3200_L"
    scene.mkdir(parents=True)
    for name in ["NOISY_RAW_010.MAT", "GT_RAW_010.MAT", "METADATA_RAW_010.MAT"]:
        (scene / name).write_text("placeholder")

    def fake_load_mat(path):
        if path.name == "METADATA_RAW_010.MAT":
            return {
                "metadata": {
                    "ISO": np.array([200]),
                    "ExposureTime": np.array([0.02]),
                    "WhiteLevel": np.array([63]),
                    "BlackLevel": np.array([0]),
                }
            }
        if path.name == "GT_RAW_010.MAT":
            return {"x": np.linspace(32, 63, 64, dtype=np.uint16).reshape(8, 8)}
        return {"x": np.arange(64, dtype=np.uint16).reshape(8, 8)}

    monkeypatch.setattr("exposure_agent.dataset.sidd_reader._load_mat", fake_load_mat)
    reader = SIDDReader(
        tmp_path / "SIDD_Small_Raw_Only",
        preview_dir=tmp_path / "previews",
        linear_rgb_dir=tmp_path / "rgb",
    )

    sample = next(reader.iter_samples())

    assert sample.scene_id == "0001_001"
    assert sample.iso == 200
    assert sample.shutter == 0.02
    assert sample.raw_gt is not None
    assert sample.image.shape == (8, 8, 3)
    assert float(sample.image.mean()) > 0.5
    assert sample.image_path is not None
    assert (tmp_path / "rgb" / "0001_001.png").exists()
    assert "metadata.ISO" in sample.metadata
    assert "metadata.ExposureTime" in sample.metadata


def test_sidd_reader_skips_damaged_samples(tmp_path, monkeypatch) -> None:
    root = tmp_path / "SIDD_Small_Raw_Only"
    (root / "Data" / "bad_folder").mkdir(parents=True)

    reader = SIDDReader(root)
    with warnings.catch_warnings(record=True) as caught:
        samples = list(reader.iter_samples())

    assert samples == []
    assert caught


def test_sidd_srgb_reader_uses_noisy_input_and_keeps_gt_reference(tmp_path) -> None:
    root = tmp_path / "SIDD_Small_sRGB_Only"
    scene = root / "Data" / "0001_001_S6_00100_00060_3200_L"
    scene.mkdir(parents=True)
    noisy = np.full((12, 16, 3), 32, dtype=np.uint8)
    gt = np.full((12, 16, 3), 224, dtype=np.uint8)
    from PIL import Image

    Image.fromarray(noisy).save(scene / "NOISY_SRGB_010.PNG")
    Image.fromarray(gt).save(scene / "GT_SRGB_010.PNG")
    reader = SIDDSRGBReader(root, preview_dir=tmp_path / "previews")

    sample = next(reader.iter_samples())

    assert sample.scene_id == "0001_001"
    assert sample.physical_scene_id == "001"
    assert float(sample.image.mean()) < 0.2
    assert sample.gt_image is None
    assert sample.gt_image_path is not None
    assert sample.metadata["source"] == "SIDD official NOISY sRGB"
    assert sample.image_path is not None
    assert (tmp_path / "previews" / "0001_001.png").exists()


def test_compact_sidd_metadata_filters_large_tiff_fields() -> None:
    compact = compact_sidd_metadata(
        {
            "metadata": {
                "ExposureTime": 1 / 60,
                "ISOSpeedRatings": 100,
                "FNumber": 1.9,
                "BlackLevel": [0, 0, 0, 0],
                "WhiteLevel": 1023,
                "AsShotNeutral": [0.5, 1.0, 0.6],
                "StripOffsets": list(range(1000)),
                "StripByteCounts": list(range(1000)),
                "UnknownTags": [{"ID": 1, "Value": list(range(1000))}],
            }
        },
        folder_name="0001_001_S6_00100_00060_3200_L",
        color_temperature=3200,
    )

    assert compact["metadata.ExposureTime"] == 1 / 60
    assert compact["metadata.ISOSpeedRatings"] == 100
    assert compact["metadata.WhiteLevel"] == 1023
    assert "metadata.StripOffsets" not in compact
    assert "metadata.StripByteCounts" not in compact
    assert "metadata.UnknownTags" not in compact


def test_prediction_writer_appends_jsonl(tmp_path) -> None:
    output = tmp_path / "outputs" / "predictions.jsonl"
    writer = PredictionWriter(output)
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=5.9)
    features = feature_bundle()
    initial = vlm_decision(metadata, target_iso=200, target_shutter=1 / 50)
    integrated = vlm_decision(metadata, target_iso=200, target_shutter=1 / 40)
    final_action = ExposureAction(target_iso=200, target_shutter_speed_s=1 / 40)
    prediction = ExposurePrediction(
        scene_id="0001_001",
        camera="S6",
        iso=100,
        shutter=1 / 60,
        ev=5.9,
        brightness_level="L",
        predicted_target_iso=200,
        predicted_target_shutter_speed_s=1 / 40,
        predicted_exposure_change_stops=1.585,
        continue_adjustment=True,
        quality_score=0.73,
        reason="test",
        fixed_features=features,
        initial_vlm_decision=initial,
        integrated_vlm_decision=integrated,
        final_action=final_action,
    )

    writer.write(prediction)

    row = json.loads(output.read_text().strip())
    assert row["scene_id"] == "0001_001"
    assert row["predicted_target_iso"] == 200

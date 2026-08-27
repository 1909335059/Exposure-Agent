from __future__ import annotations

from exposure_agent.evaluator import ImageEvaluator
import numpy as np
from PIL import Image, ImageFilter

from tests.conftest import save_solid_image


def test_evaluator_distinguishes_dark_and_bright_images(tmp_path) -> None:
    dark = save_solid_image(tmp_path / "dark.png", 8)
    bright = save_solid_image(tmp_path / "bright.png", 250)
    evaluator = ImageEvaluator()

    dark_quality = evaluator.evaluate(dark)
    bright_quality = evaluator.evaluate(bright)

    assert dark_quality.brightness < bright_quality.brightness
    assert dark_quality.shadow > bright_quality.shadow
    assert bright_quality.highlight > dark_quality.highlight


def test_darkening_a_sharp_image_does_not_create_a_severe_blur_label(tmp_path) -> None:
    grid = np.indices((64, 64)).sum(axis=0) % 2
    sharp_arr = np.uint8(grid * 255)
    sharp_rgb = np.stack([sharp_arr, sharp_arr, sharp_arr], axis=2)
    dark_rgb = np.uint8(sharp_rgb.astype(np.float32) * 0.25)
    sharp_path = tmp_path / "sharp.png"
    dark_path = tmp_path / "dark_sharp.png"
    Image.fromarray(sharp_rgb).save(sharp_path)
    Image.fromarray(dark_rgb).save(dark_path)
    evaluator = ImageEvaluator()

    sharp = evaluator.evaluate_report(sharp_path)
    dark = evaluator.evaluate_report(dark_path)

    assert abs(sharp.quality.motion_blur - dark.quality.motion_blur) < 0.15
    assert dark.quality.motion_blur < 0.75


def test_blurred_image_scores_worse_than_equal_brightness_sharp_image(tmp_path) -> None:
    grid = np.indices((64, 64)).sum(axis=0) % 2
    arr = np.uint8(grid * 255)
    rgb = np.stack([arr, arr, arr], axis=2)
    sharp_path = tmp_path / "sharp.png"
    blur_path = tmp_path / "blurred.png"
    image = Image.fromarray(rgb)
    image.save(sharp_path)
    image.filter(ImageFilter.GaussianBlur(radius=2.0)).save(blur_path)
    evaluator = ImageEvaluator()

    sharp = evaluator.evaluate_report(sharp_path)
    blurred = evaluator.evaluate_report(blur_path)

    assert blurred.quality.motion_blur > sharp.quality.motion_blur


def test_noise_metric_uses_flat_regions_and_detects_sensor_like_noise(tmp_path) -> None:
    clean = np.full((64, 64, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(7)
    noisy = np.uint8(
        np.clip(clean.astype(np.float32) + rng.normal(0, 18, clean.shape), 0, 255)
    )
    clean_path = tmp_path / "clean.png"
    noisy_path = tmp_path / "noisy.png"
    Image.fromarray(clean).save(clean_path)
    Image.fromarray(noisy).save(noisy_path)
    evaluator = ImageEvaluator()

    clean_quality = evaluator.evaluate(clean_path)
    noisy_quality = evaluator.evaluate(noisy_path)

    assert clean_quality.noise < 0.05
    assert noisy_quality.noise > clean_quality.noise

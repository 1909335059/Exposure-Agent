from __future__ import annotations

import math

from exposure_agent.models import ExposureMetadata


def exposure_scale(
    target: ExposureMetadata,
    reference: ExposureMetadata,
) -> float:
    iso_ratio = target.iso / reference.iso
    shutter_ratio = target.shutter_speed_s / reference.shutter_speed_s
    return iso_ratio * shutter_ratio


def compute_relative_ev(iso: int, shutter: float) -> float:
    if iso <= 0:
        raise ValueError("ISO must be positive")
    if shutter <= 0:
        raise ValueError("Shutter speed must be positive")
    return math.log2(1.0 / shutter) - math.log2(iso / 100.0)

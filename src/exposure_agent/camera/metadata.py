from __future__ import annotations

import json
from pathlib import Path

from exposure_agent.camera.exposure import compute_relative_ev
from exposure_agent.models import ExposureMetadata


class MetadataReader:
    def from_values(
        self,
        *,
        iso: int,
        shutter_speed_s: float,
        ev: float | None = None,
        aperture: float | None = None,
        image_id: str | None = None,
    ) -> ExposureMetadata:
        return ExposureMetadata(
            image_id=image_id,
            iso=iso,
            shutter_speed_s=shutter_speed_s,
            ev=ev if ev is not None else compute_relative_ev(iso, shutter_speed_s),
            aperture=aperture,
        )

    def from_json(self, metadata_path: str | Path) -> ExposureMetadata:
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return ExposureMetadata.model_validate(payload)

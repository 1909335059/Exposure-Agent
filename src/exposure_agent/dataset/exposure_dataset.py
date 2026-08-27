from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict

from exposure_agent.models import ExposureMetadata


class DatasetSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: Path
    metadata: ExposureMetadata


class ExposureDataset:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

    def __iter__(self) -> Iterator[DatasetSample]:
        base_dir = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                image_path = Path(payload.pop("image_path"))
                if not image_path.is_absolute():
                    image_path = base_dir / image_path
                yield DatasetSample(
                    image_path=image_path,
                    metadata=ExposureMetadata.model_validate(payload),
                )

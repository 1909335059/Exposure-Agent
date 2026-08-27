from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class PredictionWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)

    def write(self, prediction: BaseModel) -> None:
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(prediction.model_dump_json(by_alias=True))
            file.write("\n")

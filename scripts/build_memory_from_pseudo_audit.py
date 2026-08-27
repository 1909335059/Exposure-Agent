from __future__ import annotations

import argparse
import json
from pathlib import Path

from exposure_agent.models import ExposureExperience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only append-style Memory from audited search labels."
    )
    parser.add_argument("--audit-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-quality-gain", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiences: list[ExposureExperience] = []
    seen: set[str] = set()
    for line in args.audit_input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gain = float(row.get("quality_gain", 0.0))
        if row.get("dataset_split") != "train" or gain < args.min_quality_gain:
            continue
        trace = row.get("search_trace") or {}
        best = trace.get("best_candidate")
        if not isinstance(best, dict):
            continue
        run_id = (
            f"pseudo-{row['scene_id']}-"
            f"{float(row.get('source_exposure_offset_ev', 0.0)):+.2f}"
        )
        if run_id in seen:
            continue
        experience = ExposureExperience(
            run_id=run_id,
            scene_id=row["scene_id"],
            dataset_split="train",
            original_image_path=row["image_path"],
            output_image_path=best["output_image_path"],
            initial_metadata=row["metadata"],
            final_metadata=best["updated_metadata"],
            fixed_features=row["fixed_features"],
            initial_vlm_action=row["rule_action"],
            integrated_vlm_action=row["rule_action"],
            final_action=row["selected_action"],
            quality_before=row["fixed_features"]["objective_quality"],
            quality_after=best["objective_quality"],
            quality_gain=gain,
            successful=bool(best["objective_quality"].get("acceptable", False)),
            label_source="train_search_pseudo_label",
            simulator_version=str(
                trace.get("simulator_version", "exposure_simulator_v2_iso_shutter")
            ),
        )
        experiences.append(experience)
        seen.add(run_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for experience in experiences:
            output_file.write(experience.model_dump_json(by_alias=True) + "\n")
    print(f"Wrote {len(experiences)} train-only experiences to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from exposure_agent.agent import Policy
from exposure_agent.dataset import (
    ExposureSample,
    compute_relative_ev,
    load_scene_split_manifest,
    split_for_scene,
)
from exposure_agent.dataset.sidd_reader import parse_sidd_scene_name
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.optimizer import LocalSearchOptimizer
from exposure_agent.pseudo_labels import SearchPseudoLabelBuilder
from exposure_agent.training import TrainingExampleWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auditable search pseudo-labels from official SIDD sRGB images."
    )
    parser.add_argument("--srgb-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--quality-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--exposure-offsets", default="-1.0,0.0,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_scene_split_manifest(args.split_manifest)
    evaluator = ImageEvaluator(calibration_path=args.quality_calibration)
    policy = Policy()
    optimizer = LocalSearchOptimizer(
        evaluator=evaluator,
        policy=policy,
        candidate_dir=args.artifacts_dir / "search_candidates",
    )
    builder = SearchPseudoLabelBuilder(
        optimizer=optimizer,
        evaluator=evaluator,
        policy=policy,
        preview_dir=args.artifacts_dir / "previews",
    )
    offsets = tuple(
        float(token.strip()) for token in args.exposure_offsets.split(",") if token.strip()
    )
    if not offsets or any(abs(offset) > 3.0 for offset in offsets):
        raise ValueError("Exposure offsets must be numbers in the range -3..3")

    image_paths = sorted(args.srgb_root.rglob("GT_SRGB_*.PNG"))
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]
    if not image_paths:
        raise FileNotFoundError(f"No GT_SRGB_*.PNG files found under {args.srgb_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text("", encoding="utf-8")
    writer = TrainingExampleWriter(args.output)

    written = 0
    variants = 0
    with args.audit_output.open("a", encoding="utf-8") as audit_file:
        for image_path in image_paths:
            info = parse_sidd_scene_name(image_path.parent.name)
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
            sample = ExposureSample(
                image=image,
                raw_noisy=np.empty((0, 0), dtype=np.float32),
                raw_gt=None,
                iso=info.iso,
                shutter=info.shutter,
                ev=compute_relative_ev(info.iso, info.shutter),
                camera=info.camera,
                scene_id=info.scene_id,
                brightness_level=info.brightness_level,
                metadata={"source": "SIDD official GT sRGB"},
                image_path=str(image_path),
            )
            split = split_for_scene(manifest, info.scene_id)
            for example, audit in builder.build_variants(
                sample,
                dataset_split=split,
                exposure_offsets_ev=offsets,
            ):
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                if example is not None:
                    written += writer.write_examples([example])
                variants += 1

    print(
        f"Processed {len(image_paths)} images/{variants} exposure states; "
        f"wrote {written} pseudo-labels to {args.output}"
    )


if __name__ == "__main__":
    main()

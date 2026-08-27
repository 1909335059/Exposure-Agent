from __future__ import annotations

import argparse
import json
from pathlib import Path

from exposure_agent.dataset.preparation import SIDDTrainingDataPreparer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare physical-scene splits, NOISY sRGB initial labels, train-only "
            "Memory, and four-fold integration labels."
        )
    )
    parser.add_argument("--srgb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--exposure-offsets", default="-1.0,0.0,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = tuple(
        float(token.strip())
        for token in args.exposure_offsets.split(",")
        if token.strip()
    )
    summary = SIDDTrainingDataPreparer(
        srgb_root=args.srgb_root,
        output_dir=args.output_dir,
        seed=args.seed,
        folds=args.folds,
        exposure_offsets_ev=offsets,
        max_samples=args.max_samples,
        progress=print,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

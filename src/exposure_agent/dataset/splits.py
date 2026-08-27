from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Literal

DatasetSplit = Literal["train", "validation", "test"]


def build_scene_split_manifest(
    scene_ids: list[str],
    output_path: str | Path,
    *,
    seed: int = 42,
    physical_scene_ids: list[str] | None = None,
) -> dict:
    """Build a deterministic split, optionally grouping SIDD instances by scene."""
    unique_samples = sorted(set(scene_ids))
    if not unique_samples:
        raise ValueError("Cannot split an empty scene list")
    sample_to_group = _sample_group_map(scene_ids, physical_scene_ids)
    groups = sorted(set(sample_to_group.values()))
    random.Random(seed).shuffle(groups)
    train_count, validation_count = _split_counts(len(groups))

    group_assignments: dict[str, DatasetSplit] = {}
    for index, group_id in enumerate(groups):
        if index < train_count:
            group_assignments[group_id] = "train"
        elif index < train_count + validation_count:
            group_assignments[group_id] = "validation"
        else:
            group_assignments[group_id] = "test"
    assignments = {
        scene_id: group_assignments[sample_to_group[scene_id]]
        for scene_id in unique_samples
    }
    grouped_samples = {
        group_id: sorted(
            scene_id
            for scene_id, sample_group in sample_to_group.items()
            if sample_group == group_id
        )
        for group_id in sorted(group_assignments)
    }
    payload = {
        "version": 2,
        "seed": seed,
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "group_key": (
            "physical_scene_id" if physical_scene_ids is not None else "scene_id"
        ),
        "group_counts": _assignment_counts(group_assignments),
        "sample_counts": _assignment_counts(assignments),
        "counts": _assignment_counts(assignments),
        "group_assignments": dict(sorted(group_assignments.items())),
        "sample_to_group": dict(sorted(sample_to_group.items())),
        "groups": grouped_samples,
        "assignments": dict(sorted(assignments.items())),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_crossfold_manifest(
    split_manifest: dict,
    output_path: str | Path,
    *,
    folds: int = 4,
    seed: int = 42,
) -> dict:
    """Assign train-only physical groups to held-out query folds."""
    group_assignments = split_manifest.get("group_assignments")
    if not isinstance(group_assignments, dict):
        raise ValueError("Cross-fold generation requires a version 2 grouped split")
    train_groups = sorted(
        group_id
        for group_id, split in group_assignments.items()
        if split == "train"
    )
    if folds < 2 or len(train_groups) < folds:
        raise ValueError("Cross-fold generation requires at least one train group per fold")
    random.Random(seed).shuffle(train_groups)
    query_groups = [train_groups[index::folds] for index in range(folds)]
    fold_rows = []
    group_to_fold: dict[str, int] = {}
    for fold_index, held_out in enumerate(query_groups):
        held_out_set = set(held_out)
        for group_id in held_out:
            group_to_fold[group_id] = fold_index
        fold_rows.append(
            {
                "fold": fold_index,
                "query_groups": sorted(held_out),
                "memory_groups": sorted(set(train_groups) - held_out_set),
            }
        )
    payload = {
        "version": 1,
        "seed": seed,
        "fold_count": folds,
        "source_split_seed": split_manifest.get("seed"),
        "group_key": split_manifest.get("group_key"),
        "train_groups": sorted(train_groups),
        "group_to_fold": dict(sorted(group_to_fold.items())),
        "folds": fold_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_scene_split_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Split manifest must contain an assignments object")
    invalid = set(assignments.values()) - {"train", "validation", "test"}
    if invalid:
        raise ValueError(f"Invalid split values: {sorted(invalid)}")
    return payload


def ensure_scene_split_manifest(
    scene_ids: list[str],
    path: str | Path,
    *,
    seed: int = 42,
    physical_scene_ids: list[str] | None = None,
) -> dict:
    manifest_path = Path(path)
    if manifest_path.exists():
        payload = load_scene_split_manifest(manifest_path)
        missing = sorted(set(scene_ids) - set(payload["assignments"]))
        expected_key = "physical_scene_id" if physical_scene_ids is not None else "scene_id"
        incompatible = payload.get("group_key", "scene_id") != expected_key
        if not missing and not incompatible:
            return payload
    return build_scene_split_manifest(
        scene_ids,
        manifest_path,
        seed=seed,
        physical_scene_ids=physical_scene_ids,
    )


def split_for_scene(
    manifest: dict,
    scene_id: str,
    physical_scene_id: str | None = None,
) -> DatasetSplit:
    assignments = manifest["assignments"]
    if scene_id in assignments:
        return assignments[scene_id]
    group_assignments = manifest.get("group_assignments", {})
    if physical_scene_id is not None and physical_scene_id in group_assignments:
        return group_assignments[physical_scene_id]
    raise KeyError(f"Scene {scene_id} is not present in the split manifest")


def _sample_group_map(
    scene_ids: list[str],
    physical_scene_ids: list[str] | None,
) -> dict[str, str]:
    if physical_scene_ids is None:
        return {scene_id: scene_id for scene_id in scene_ids}
    if len(scene_ids) != len(physical_scene_ids):
        raise ValueError("scene_ids and physical_scene_ids must have equal length")
    mapping: dict[str, str] = {}
    for scene_id, physical_scene_id in zip(scene_ids, physical_scene_ids):
        existing = mapping.get(scene_id)
        if existing is not None and existing != physical_scene_id:
            raise ValueError(f"Scene {scene_id} maps to multiple physical scenes")
        mapping[scene_id] = physical_scene_id
    return mapping


def _split_counts(count: int) -> tuple[int, int]:
    if count == 1:
        return 1, 0
    if count == 2:
        return 1, 1
    validation_count = max(1, round(count * 0.10))
    test_count = max(1, round(count * 0.10))
    train_count = count - validation_count - test_count
    if train_count < 1:
        train_count = 1
        validation_count = count - train_count - test_count
    return train_count, validation_count


def _assignment_counts(assignments: dict[str, DatasetSplit]) -> dict[str, int]:
    return {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation", "test")
    }

from __future__ import annotations

from exposure_agent.dataset import (
    build_crossfold_manifest,
    build_scene_split_manifest,
    load_scene_split_manifest,
)


def test_scene_split_is_reproducible_and_disjoint(tmp_path) -> None:
    scene_ids = [f"{index:04d}_001" for index in range(20)]
    first = build_scene_split_manifest(scene_ids, tmp_path / "first.json", seed=42)
    second = build_scene_split_manifest(scene_ids, tmp_path / "second.json", seed=42)

    assert first["assignments"] == second["assignments"]
    assert first["counts"] == {"train": 16, "validation": 2, "test": 2}
    assert set(first["assignments"]) == set(scene_ids)
    assert load_scene_split_manifest(tmp_path / "first.json") == first


def test_sidd_split_groups_all_instances_by_physical_scene(tmp_path) -> None:
    scene_ids = [
        f"{group * 2 + instance:04d}_{group + 1:03d}"
        for group in range(10)
        for instance in range(2)
    ]
    physical_ids = [f"{group + 1:03d}" for group in range(10) for _ in range(2)]

    manifest = build_scene_split_manifest(
        scene_ids,
        tmp_path / "physical.json",
        seed=42,
        physical_scene_ids=physical_ids,
    )

    assert manifest["group_key"] == "physical_scene_id"
    assert manifest["group_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert manifest["sample_counts"] == {"train": 16, "validation": 2, "test": 2}
    for physical_id, grouped_scene_ids in manifest["groups"].items():
        assert {
            manifest["assignments"][scene_id] for scene_id in grouped_scene_ids
        } == {manifest["group_assignments"][physical_id]}


def test_crossfold_memory_never_contains_query_physical_scene(tmp_path) -> None:
    scene_ids = [f"{index:04d}_{index + 1:03d}" for index in range(10)]
    physical_ids = [f"{index + 1:03d}" for index in range(10)]
    split = build_scene_split_manifest(
        scene_ids,
        tmp_path / "split.json",
        seed=42,
        physical_scene_ids=physical_ids,
    )

    folds = build_crossfold_manifest(split, tmp_path / "folds.json", seed=42)

    query_groups = []
    for fold in folds["folds"]:
        assert set(fold["query_groups"]).isdisjoint(fold["memory_groups"])
        assert len(fold["query_groups"]) == 2
        assert len(fold["memory_groups"]) == 6
        query_groups.extend(fold["query_groups"])
    assert sorted(query_groups) == sorted(folds["train_groups"])

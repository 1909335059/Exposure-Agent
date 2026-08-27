from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from exposure_agent.agent import Policy
from exposure_agent.dataset.sidd_reader import SIDDSRGBReader
from exposure_agent.dataset.splits import (
    build_crossfold_manifest,
    build_scene_split_manifest,
    split_for_scene,
)
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.memory import JsonlMemory
from exposure_agent.models import ExposureExperience
from exposure_agent.optimizer import LocalSearchConfig, LocalSearchOptimizer
from exposure_agent.pseudo_labels import (
    CrossFoldIntegrationBuilder,
    SearchPseudoLabelBuilder,
    experience_from_pseudo_audit,
)
from exposure_agent.training import TrainingExampleWriter


@dataclass(frozen=True)
class SIDDPreparationPaths:
    root: Path
    split_manifest: Path
    crossfold_manifest: Path
    quality_calibration: Path
    initial_sft: Path
    initial_audit: Path
    integration_sft: Path
    integration_audit: Path
    combined_sft: Path
    train_memory: Path
    fold_memory_dir: Path
    previews_dir: Path
    artifacts_dir: Path
    summary: Path

    @classmethod
    def under(cls, root: str | Path) -> "SIDDPreparationPaths":
        base = Path(root)
        return cls(
            root=base,
            split_manifest=base / "sidd_physical_scene_splits.json",
            crossfold_manifest=base / "sidd_train_crossfolds.json",
            quality_calibration=base / "quality_calibration_train_gt.json",
            initial_sft=base / "initial_sft.jsonl",
            initial_audit=base / "initial_audit.jsonl",
            integration_sft=base / "integration_sft.jsonl",
            integration_audit=base / "integration_audit.jsonl",
            combined_sft=base / "combined_sft.jsonl",
            train_memory=base / "memory_train.jsonl",
            fold_memory_dir=base / "fold_memories",
            previews_dir=base / "previews_noisy_srgb",
            artifacts_dir=base / "artifacts",
            summary=base / "preparation_summary.json",
        )


class SIDDTrainingDataPreparer:
    """Builds SIDD splits, initial labels, train Memory, and integration labels."""

    def __init__(
        self,
        *,
        srgb_root: str | Path,
        output_dir: str | Path,
        seed: int = 42,
        folds: int = 4,
        exposure_offsets_ev: tuple[float, ...] = (-1.0, 0.0, 1.0),
        max_samples: int | None = None,
        preview_max_dimension: int = 1024,
        min_quality_gain: float = 0.02,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.srgb_root = Path(srgb_root)
        self.paths = SIDDPreparationPaths.under(output_dir)
        self.seed = seed
        self.folds = folds
        self.exposure_offsets_ev = exposure_offsets_ev
        self.max_samples = max_samples
        self.preview_max_dimension = preview_max_dimension
        self.min_quality_gain = min_quality_gain
        self.progress = progress or (lambda _: None)

    def run(self) -> dict:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        reader = SIDDSRGBReader(
            self.srgb_root,
            preview_dir=self.paths.previews_dir,
            preview_max_dimension=self.preview_max_dimension,
        )
        samples = list(reader.iter_samples())
        if self.max_samples is not None:
            samples = _stratified_sample_limit(samples, self.max_samples)
        if not samples:
            raise ValueError("No valid SIDD NOISY sRGB samples were found")
        self._validate_samples(samples)

        scene_ids = [sample.scene_id for sample in samples]
        physical_ids = [str(sample.physical_scene_id) for sample in samples]
        split_manifest = build_scene_split_manifest(
            scene_ids,
            self.paths.split_manifest,
            seed=self.seed,
            physical_scene_ids=physical_ids,
        )
        crossfold_manifest = build_crossfold_manifest(
            split_manifest,
            self.paths.crossfold_manifest,
            folds=self.folds,
            seed=self.seed,
        )
        self.progress(
            "Loaded SIDD samples and built physical-scene split: "
            f"{split_manifest['sample_counts']}"
        )

        train_gt_paths = [
            sample.gt_image_path
            for sample in samples
            if split_for_scene(
                split_manifest,
                sample.scene_id,
                sample.physical_scene_id,
            )
            == "train"
            and sample.gt_image_path is not None
        ]
        calibration = ImageEvaluator.calibrate(
            train_gt_paths,
            self.paths.quality_calibration,
        )
        evaluator = ImageEvaluator(calibration_path=self.paths.quality_calibration)
        policy = Policy()

        initial_optimizer = LocalSearchOptimizer(
            evaluator=evaluator,
            policy=policy,
            config=LocalSearchConfig(retain_candidate_images=False),
            candidate_dir=self.paths.artifacts_dir / "initial_search",
        )
        initial_builder = SearchPseudoLabelBuilder(
            optimizer=initial_optimizer,
            evaluator=evaluator,
            policy=policy,
            preview_dir=self.paths.previews_dir,
            min_quality_gain=self.min_quality_gain,
        )
        initial_rows = self._build_initial_rows(
            samples=samples,
            split_manifest=split_manifest,
            builder=initial_builder,
        )
        self.progress(
            f"Built {len(initial_rows)} auditable initial exposure states from NOISY sRGB"
        )

        train_experiences = self._build_train_memory(initial_rows)
        self._write_experiences(self.paths.train_memory, train_experiences)
        self.progress(
            f"Built append-only train Memory with {len(train_experiences)} positive experiences"
        )

        integration_count, integration_valid = self._build_integration_rows(
            initial_rows=initial_rows,
            crossfold_manifest=crossfold_manifest,
            train_experiences=train_experiences,
            evaluator=evaluator,
            policy=policy,
        )
        self._combine_sft_files()
        summary = {
            "version": 1,
            "source": "SIDD Small official NOISY sRGB inputs",
            "gt_usage": "train-split quality calibration only",
            "seed": self.seed,
            "folds": self.folds,
            "exposure_offsets_ev": list(self.exposure_offsets_ev),
            "sample_count": len(samples),
            "physical_scene_count": len(set(physical_ids)),
            "split_group_counts": split_manifest["group_counts"],
            "split_sample_counts": split_manifest["sample_counts"],
            "quality_calibration": calibration,
            "initial_state_count": len(initial_rows),
            "initial_label_count": _jsonl_count(self.paths.initial_sft),
            "train_memory_experience_count": len(train_experiences),
            "integration_state_count": integration_count,
            "integration_label_count": integration_valid,
            "combined_label_count": _jsonl_count(self.paths.combined_sft),
            "paths": {
                key: str(value)
                for key, value in self.paths.__dict__.items()
                if key != "root"
            },
        }
        self.paths.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _build_initial_rows(self, *, samples, split_manifest: dict, builder) -> list[dict]:
        _truncate(self.paths.initial_sft)
        _truncate(self.paths.initial_audit)
        writer = TrainingExampleWriter(self.paths.initial_sft)
        rows: list[dict] = []
        with self.paths.initial_audit.open("a", encoding="utf-8") as audit_file:
            for index, sample in enumerate(samples, start=1):
                split = split_for_scene(
                    split_manifest,
                    sample.scene_id,
                    sample.physical_scene_id,
                )
                for example, audit in builder.build_variants(
                    sample,
                    dataset_split=split,
                    exposure_offsets_ev=self.exposure_offsets_ev,
                ):
                    audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    rows.append(audit)
                    if example is not None:
                        writer.write_examples([example])
                if index % 20 == 0 or index == len(samples):
                    self.progress(f"Initial labels: processed {index}/{len(samples)} samples")
        return rows

    def _build_train_memory(self, rows: list[dict]) -> list[ExposureExperience]:
        experiences: list[ExposureExperience] = []
        seen: set[str] = set()
        for row in rows:
            experience = experience_from_pseudo_audit(
                row,
                min_quality_gain=self.min_quality_gain,
            )
            if experience is None or experience.run_id in seen:
                continue
            experiences.append(experience)
            seen.add(experience.run_id)
        return experiences

    def _build_integration_rows(
        self,
        *,
        initial_rows: list[dict],
        crossfold_manifest: dict,
        train_experiences: list[ExposureExperience],
        evaluator: ImageEvaluator,
        policy: Policy,
    ) -> tuple[int, int]:
        _truncate(self.paths.integration_sft)
        _truncate(self.paths.integration_audit)
        writer = TrainingExampleWriter(self.paths.integration_sft)
        total = 0
        valid = 0
        fold_memories: dict[int, JsonlMemory] = {}
        for fold in crossfold_manifest["folds"]:
            fold_index = int(fold["fold"])
            allowed = set(fold["memory_groups"])
            experiences = [
                experience
                for experience in train_experiences
                if experience.physical_scene_id in allowed
            ]
            memory_path = self.paths.fold_memory_dir / f"fold_{fold_index}.jsonl"
            self._write_experiences(memory_path, experiences)
            fold_memories[fold_index] = JsonlMemory(
                memory_path,
                top_k=3,
                min_quality_gain=self.min_quality_gain,
                read_only=True,
            )

        full_memory = JsonlMemory(
            self.paths.train_memory,
            top_k=3,
            min_quality_gain=self.min_quality_gain,
            read_only=True,
        )
        builders: dict[int | None, CrossFoldIntegrationBuilder] = {}

        with self.paths.integration_audit.open("a", encoding="utf-8") as audit_file:
            for row in initial_rows:
                split = row.get("dataset_split")
                physical_scene_id = row.get("physical_scene_id")
                if split == "train":
                    fold_index = int(
                        crossfold_manifest["group_to_fold"][physical_scene_id]
                    )
                    memory = fold_memories[fold_index]
                else:
                    fold_index = None
                    memory = full_memory
                if fold_index not in builders:
                    optimizer = LocalSearchOptimizer(
                        evaluator=evaluator,
                        policy=policy,
                        config=LocalSearchConfig(retain_candidate_images=False),
                        candidate_dir=(
                            self.paths.artifacts_dir
                            / "integration_search"
                            / (f"fold_{fold_index}" if fold_index is not None else "heldout")
                        ),
                    )
                    builders[fold_index] = CrossFoldIntegrationBuilder(
                        optimizer=optimizer,
                        memory=memory,
                        policy=policy,
                        min_quality_gain=self.min_quality_gain,
                    )
                example, audit = builders[fold_index].build_from_audit(
                    row,
                    cross_fold=fold_index,
                )
                self._validate_retrieval_isolation(audit)
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                total += 1
                if example is not None:
                    valid += writer.write_examples([example])
                if total % 50 == 0 or total == len(initial_rows):
                    self.progress(
                        f"Integration labels: processed {total}/{len(initial_rows)} states"
                    )
        return total, valid

    def _combine_sft_files(self) -> None:
        _truncate(self.paths.combined_sft)
        seen: set[str] = set()
        with self.paths.combined_sft.open("a", encoding="utf-8") as output:
            for path in (self.paths.initial_sft, self.paths.integration_sft):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    example_id = str(row["example_id"])
                    if example_id in seen:
                        continue
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen.add(example_id)

    @staticmethod
    def _write_experiences(
        path: Path,
        experiences: list[ExposureExperience],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            for experience in experiences:
                output.write(experience.model_dump_json(by_alias=True) + "\n")

    @staticmethod
    def _validate_samples(samples) -> None:
        scene_ids = [sample.scene_id for sample in samples]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("SIDD sample IDs are not unique")
        if any(sample.physical_scene_id is None for sample in samples):
            raise ValueError("Every SIDD sample must have a physical_scene_id")
        if any(sample.gt_image_path is None for sample in samples):
            raise ValueError("Every SIDD sample must have a paired GT sRGB calibration path")

    @staticmethod
    def _validate_retrieval_isolation(audit: dict) -> None:
        context = audit.get("memory_context")
        if not isinstance(context, dict):
            return
        query_physical = audit.get("physical_scene_id")
        for example in context.get("examples", []):
            if example.get("physical_scene_id") == query_physical:
                raise ValueError(
                    "Physical-scene leakage detected in integration retrieval: "
                    f"{query_physical}"
                )


def _truncate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _stratified_sample_limit(samples, limit: int):
    if limit < 1:
        raise ValueError("max_samples must be positive")
    if limit >= len(samples):
        return samples
    groups: dict[str, list] = {}
    for sample in samples:
        groups.setdefault(str(sample.physical_scene_id), []).append(sample)
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for group_id in sorted(groups):
            group = groups[group_id]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected, key=lambda sample: sample.scene_id)

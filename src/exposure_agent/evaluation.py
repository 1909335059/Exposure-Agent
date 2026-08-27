from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


QUALITY_FIELDS = (
    "brightness",
    "noise",
    "motion_blur",
    "highlight",
    "shadow",
    "overall_quality",
)


def evaluate_action_predictions(
    *,
    predictions_path: str | Path,
    targets_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    targets = {
        row["scene_id"]: _target_from_sft(row)
        for row in _read_jsonl(Path(targets_path))
        if row.get("scene_id") and row.get("dataset_split", "train") == split
    }
    predictions = {
        row.get("scene_id"): row
        for row in _read_jsonl(Path(predictions_path))
        if row.get("scene_id")
    }
    matched = [
        (scene_id, target, predictions[scene_id])
        for scene_id, target in targets.items()
        if target is not None and scene_id in predictions
    ]
    quality_errors = {field: [] for field in QUALITY_FIELDS}
    iso_errors: list[float] = []
    shutter_errors: list[float] = []
    true_continue: list[bool] = []
    predicted_continue: list[bool] = []
    for _, target, prediction in matched:
        predicted_quality = prediction.get("quality") or {}
        for field in QUALITY_FIELDS:
            if target["quality"].get(field) is not None and predicted_quality.get(field) is not None:
                quality_errors[field].append(
                    abs(float(predicted_quality[field]) - float(target["quality"][field]))
                )
        target_action = target["action"]
        iso_errors.append(
            abs(float(prediction["predicted_target_iso"]) - float(target_action["ISO"]))
        )
        shutter_errors.append(
            abs(
                math.log2(
                    max(float(prediction["predicted_target_shutter_speed_s"]), 1e-8)
                )
                - math.log2(max(float(target_action["Shutter"]), 1e-8))
            )
        )
        true_continue.append(bool(target["continue"]))
        predicted_continue.append(bool(prediction["continue_adjustment"]))
    return {
        "split": split,
        "target_count": len(targets),
        "matched_count": len(matched),
        "json_validity_rate": len(matched) / len(targets) if targets else 0.0,
        "quality_mae": {
            field: mean(values) if values else None
            for field, values in quality_errors.items()
        },
        "target_iso_mae": mean(iso_errors) if iso_errors else None,
        "target_shutter_log2_seconds_mae": (
            mean(shutter_errors) if shutter_errors else None
        ),
        "continue": _binary_metrics(true_continue, predicted_continue),
    }


def evaluate_agent_runs(
    *,
    runs_path: str | Path,
    split: str | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(Path(runs_path))
    if split is not None:
        rows = [row for row in rows if row.get("dataset_split", "train") == split]
    gains: list[float] = []
    iterations: list[int] = []
    iso_changes: list[float] = []
    shutter_stop_changes: list[float] = []
    successes = 0
    max_round_failures = 0
    for row in rows:
        rounds = row.get("iterations") or []
        if not rounds:
            continue
        first = rounds[0]
        initial_report = first.get("objective_quality_before") or {}
        final_report = row.get("final_objective_quality") or {}
        initial_overall = (initial_report.get("quality") or {}).get("overall_quality")
        final_overall = (final_report.get("quality") or {}).get("overall_quality")
        if initial_overall is not None and final_overall is not None:
            gains.append(float(final_overall) - float(initial_overall))
        if final_report.get("acceptable") is True:
            successes += 1
        if row.get("stop_reason") == "max_rounds_best_result" and final_report.get(
            "acceptable"
        ) is not True:
            max_round_failures += 1
        iterations.append(len(rounds))
        initial_metadata = row.get("initial_metadata") or {}
        final_metadata = row.get("final_metadata") or {}
        if initial_metadata and final_metadata:
            iso_changes.append(
                abs(float(final_metadata.get("iso", 0)) - float(initial_metadata.get("iso", 0)))
            )
            initial_shutter = max(float(initial_metadata.get("shutter_speed_s", 1)), 1e-8)
            final_shutter = max(float(final_metadata.get("shutter_speed_s", 1)), 1e-8)
            shutter_stop_changes.append(abs(math.log2(final_shutter / initial_shutter)))
    count = len(rows)
    return {
        "split": split,
        "run_count": count,
        "mean_objective_quality_gain": mean(gains) if gains else None,
        "success_rate": successes / count if count else 0.0,
        "mean_iterations": mean(iterations) if iterations else None,
        "unsatisfied_after_max_rounds_rate": (
            max_round_failures / count if count else 0.0
        ),
        "mean_absolute_iso_change": mean(iso_changes) if iso_changes else None,
        "mean_absolute_shutter_stop_change": (
            mean(shutter_stop_changes) if shutter_stop_changes else None
        ),
    }


def _target_from_sft(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None
    try:
        payload = json.loads(assistant["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _binary_metrics(targets: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(t and p for t, p in zip(targets, predictions))
    fp = sum(not t and p for t, p in zip(targets, predictions))
    fn = sum(t and not p for t, p in zip(targets, predictions))
    tn = sum(not t and not p for t, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image

from exposure_agent.agent import ExposureAgent, Policy
from exposure_agent.camera import MetadataReader
from exposure_agent.config import get_settings
from exposure_agent.dataset import (
    PredictionWriter,
    SIDDReader,
    build_scene_split_manifest,
    ensure_scene_split_manifest,
    load_scene_split_manifest,
    rgb_to_srgb,
    save_rgb_png,
    split_for_scene,
)
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.evaluation import evaluate_action_predictions, evaluate_agent_runs
from exposure_agent.memory import JsonlMemory, NoOpMemory
from exposure_agent.models import ExposurePrediction
from exposure_agent.optimizer import LocalSearchOptimizer, NoOpOptimizer
from exposure_agent.reporting import StructuredReportWriter
from exposure_agent.pseudo_labels import SearchPseudoLabelBuilder
from exposure_agent.dataset.preparation import SIDDTrainingDataPreparer
from exposure_agent.training import TrainingExampleWriter, export_training_from_memory
from exposure_agent.vlm import build_vlm_client

app = typer.Typer(help="VLM-guided exposure action prediction and research loop.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Dataset type. Currently supports: sidd."),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data_root", help="Dataset root path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Prediction JSONL output path."),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max_samples", help="Maximum number of samples to process."),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="single or loop. loop runs the full research agent."),
    ] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option("--max_iterations", help="Closed-loop round limit; capped at 3."),
    ] = None,
    memory_path: Annotated[
        Path | None,
        typer.Option("--memory_path", help="JSONL memory/RAG store path."),
    ] = None,
    training_output: Annotated[
        Path | None,
        typer.Option("--training_output", help="Qwen-style SFT JSONL output path."),
    ] = None,
    structured_output: Annotated[
        Path | None,
        typer.Option(
            "--structured_output",
            help="Human-readable structured loop report JSON path.",
        ),
    ] = None,
    markdown_output: Annotated[
        Path | None,
        typer.Option("--markdown_output", help="Chinese Markdown loop report path."),
    ] = None,
    split_manifest: Annotated[
        Path | None,
        typer.Option("--split_manifest", help="Scene-level train/validation/test manifest."),
    ] = None,
    no_rag: Annotated[
        bool,
        typer.Option("--no_rag", help="Disable JSONL Memory/RAG retrieval and writeback."),
    ] = False,
    no_local_search: Annotated[
        bool,
        typer.Option("--no_local_search", help="Disable local grid search refinement."),
    ] = False,
    save_rgb_intermediate: Annotated[
        bool,
        typer.Option(
            "--save_rgb_intermediate",
            help="Save linear RGB intermediates as PNG files for RGB->sRGB testing.",
        ),
    ] = False,
    rgb_output_dir: Annotated[
        Path | None,
        typer.Option("--rgb_output_dir", help="Directory for linear RGB PNG outputs."),
    ] = None,
) -> None:
    """Run dataset inference when --dataset is provided."""
    if ctx.invoked_subcommand is not None:
        return
    if dataset is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    if dataset.lower() != "sidd":
        raise typer.BadParameter(f"Unsupported dataset: {dataset}")

    settings = get_settings()
    resolved_mode = (mode or settings.run_mode).lower()
    if resolved_mode not in {"single", "loop"}:
        raise typer.BadParameter("--mode must be single or loop")
    resolved_data_root = data_root or settings.sidd_data_root
    resolved_output = output or settings.predictions_output
    resolved_memory_path = memory_path or settings.memory_path
    resolved_training_output = training_output or settings.training_output
    resolved_split_manifest = (
        split_manifest
        or Path(resolved_output).parent / settings.split_manifest_path.name
    )
    reader = SIDDReader(
        resolved_data_root,
        preview_dir=Path(resolved_output).parent / "previews",
        linear_rgb_dir=(
            rgb_output_dir
            if rgb_output_dir is not None
            else Path(resolved_output).parent / "rgb"
            if save_rgb_intermediate
            else None
        ),
    )
    split_payload = ensure_scene_split_manifest(
        reader.scene_ids(),
        resolved_split_manifest,
        seed=42,
        physical_scene_ids=reader.physical_scene_ids(),
    )
    agent = _build_agent(
        mode=resolved_mode,
        memory_path=resolved_memory_path,
        use_rag=settings.enable_rag and not no_rag,
        use_local_search=settings.enable_local_search and not no_local_search,
    )
    writer = PredictionWriter(resolved_output)
    training_writer = (
        TrainingExampleWriter(resolved_training_output)
        if resolved_mode == "loop"
        else None
    )
    structured_writer = (
        StructuredReportWriter(
            structured_output or Path(resolved_output).parent / "structured_report.json",
            markdown_output_path=markdown_output,
        )
        if resolved_mode == "loop"
        else None
    )
    loop_iterations = min(max_iterations or settings.max_iterations, 3)

    count = 0
    for sample in reader.iter_samples(max_samples=max_samples):
        dataset_split = split_for_scene(
            split_payload,
            sample.scene_id,
            sample.physical_scene_id,
        )
        if resolved_mode == "loop":
            result = agent.run_closed_loop_sample(
                sample,
                max_iterations=loop_iterations,
                dataset_split=dataset_split,
            )
            writer.write(result)
            if training_writer is not None:
                training_writer.write_result(result)
            if structured_writer is not None:
                structured_writer.add_result(result)
        else:
            prediction = agent.predict_sample(sample, dataset_split=dataset_split)
            writer.write(prediction)
        count += 1
    if structured_writer is not None:
        structured_writer.close()
    if resolved_mode == "single":
        typer.echo(f"Wrote {count} predictions to {resolved_output}")
    else:
        typer.echo(f"Wrote {count} loop results to {resolved_output}")


@app.command()
def run(
    image: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    scene_id: Annotated[
        str | None,
        typer.Option("--scene_id", help="Optional stable scene ID for leakage exclusion."),
    ] = None,
    dataset_split: Annotated[
        str,
        typer.Option("--dataset_split", help="train, validation, or test."),
    ] = "train",
    iso: Annotated[int | None, typer.Option(help="Current ISO value.")] = None,
    shutter: Annotated[
        float | None,
        typer.Option("--shutter", help="Current shutter speed in seconds."),
    ] = None,
    ev: Annotated[
        float | None,
        typer.Option(help="Optional derived EV metadata; calculated from ISO/shutter if omitted."),
    ] = None,
    aperture: Annotated[
        float | None,
        typer.Option(help="Optional aperture value."),
    ] = None,
    metadata_json: Annotated[
        Path | None,
        typer.Option("--metadata-json", help="Optional JSON metadata file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSONL output file."),
    ] = None,
    structured_output: Annotated[
        Path | None,
        typer.Option(
            "--structured_output",
            help="Optional structured loop report JSON file.",
        ),
    ] = None,
    markdown_output: Annotated[
        Path | None,
        typer.Option("--markdown_output", help="Optional Chinese Markdown loop report."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="single or loop."),
    ] = "single",
    max_iterations: Annotated[
        int | None,
        typer.Option("--max_iterations", help="Closed-loop round limit; capped at 3."),
    ] = None,
) -> None:
    settings = get_settings()
    reader = MetadataReader()
    metadata = (
        reader.from_json(metadata_json)
        if metadata_json is not None
        else _metadata_from_cli_values(
            reader=reader,
            image=image,
            iso=iso,
            shutter=shutter,
            ev=ev,
            aperture=aperture,
        )
    )
    resolved_mode = mode.lower()
    if resolved_mode not in {"single", "loop"}:
        raise typer.BadParameter("--mode must be single or loop")
    if dataset_split not in {"train", "validation", "test"}:
        raise typer.BadParameter("--dataset_split must be train, validation, or test")
    agent = _build_agent(
        mode=resolved_mode,
        memory_path=settings.memory_path,
        use_rag=settings.enable_rag,
        use_local_search=settings.enable_local_search,
    )
    result = (
        agent.run_closed_loop_image(
            image_path=image,
            metadata=metadata,
            max_iterations=max_iterations or settings.max_iterations,
            scene_id=scene_id or metadata.image_id,
            dataset_split=dataset_split,
        )
        if resolved_mode == "loop"
        else agent.predict_image(
            image_path=image,
            metadata=metadata,
            scene_id=scene_id or metadata.image_id,
            dataset_split=dataset_split,
        )
    )
    if structured_output is not None:
        if resolved_mode != "loop":
            raise typer.BadParameter("--structured_output requires --mode loop")
        structured_writer = StructuredReportWriter(
            structured_output,
            markdown_output_path=markdown_output,
        )
        structured_writer.add_result(result)
        structured_writer.close()
    if output is not None:
        PredictionWriter(output).write(result)
    typer.echo(result.model_dump_json(indent=2))


@app.command("rgb-to-srgb")
def convert_rgb_to_srgb(
    rgb_image: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    output_png: Annotated[
        Path,
        typer.Argument(file_okay=True, dir_okay=False),
    ],
) -> None:
    """Convert a saved linear RGB image to an sRGB PNG."""
    rgb = np.asarray(Image.open(rgb_image).convert("RGB"), dtype=np.float32) / 255.0
    srgb = rgb_to_srgb(rgb)
    save_rgb_png(srgb, output_png)
    typer.echo(f"Wrote sRGB preview to {output_png}")


@app.command("export-training")
def export_training(
    memory_path: Annotated[
        Path,
        typer.Option("--memory_path", help="Input JSONL memory path."),
    ] = Path("outputs/memory.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Qwen-style SFT JSONL path."),
    ] = Path("outputs/training_sft.jsonl"),
) -> None:
    count = export_training_from_memory(memory_path=memory_path, output_path=output)
    typer.echo(f"Wrote {count} training examples to {output}")


@app.command("build-splits")
def build_splits_command(
    data_root: Annotated[
        Path | None,
        typer.Option("--data_root", help="SIDD Small Raw Only root."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output scene split manifest."),
    ] = Path("outputs/sidd_scene_splits.json"),
    seed: Annotated[int, typer.Option("--seed", help="Deterministic split seed.")] = 42,
) -> None:
    settings = get_settings()
    reader = SIDDReader(data_root or settings.sidd_data_root)
    payload = build_scene_split_manifest(
        reader.scene_ids(),
        output,
        seed=seed,
        physical_scene_ids=reader.physical_scene_ids(),
    )
    typer.echo(
        f"Wrote scene split manifest to {output}: "
        f"{json.dumps(payload['counts'], ensure_ascii=False)}"
    )


@app.command("prepare-sidd-training")
def prepare_sidd_training_command(
    srgb_root: Annotated[
        Path,
        typer.Option(
            "--srgb_root",
            help="SIDD Small sRGB root containing official NOISY/GT pairs.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output_dir", help="Output directory for manifests and SFT data."),
    ] = Path("outputs/sidd_training"),
    max_samples: Annotated[
        int | None,
        typer.Option("--max_samples", help="Optional local smoke-test limit."),
    ] = None,
    exposure_offsets: Annotated[
        str,
        typer.Option(
            "--exposure_offsets",
            help="Comma-separated source exposure offsets in stops.",
        ),
    ] = "-1.0,0.0,1.0",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    folds: Annotated[int, typer.Option("--folds")] = 4,
) -> None:
    try:
        offsets = tuple(
            float(token.strip())
            for token in exposure_offsets.split(",")
            if token.strip()
        )
    except ValueError as exc:
        raise typer.BadParameter("--exposure_offsets must contain numbers") from exc
    if not offsets or any(abs(offset) > 3.0 for offset in offsets):
        raise typer.BadParameter("Provide exposure offsets in the range -3..3")
    summary = SIDDTrainingDataPreparer(
        srgb_root=srgb_root,
        output_dir=output_dir,
        seed=seed,
        folds=folds,
        exposure_offsets_ev=offsets,
        max_samples=max_samples,
        progress=typer.echo,
    ).run()
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command("calibrate-quality")
def calibrate_quality_command(
    srgb_root: Annotated[
        Path,
        typer.Option("--srgb_root", help="SIDD Small sRGB Only root."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Quality calibration JSON path."),
    ] = Path("outputs/quality_calibration.json"),
    split_manifest: Annotated[
        Path | None,
        typer.Option("--split_manifest", help="Scene manifest; only train GT images are used."),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max_samples", help="Optional calibration image limit."),
    ] = None,
) -> None:
    settings = get_settings()
    manifest_path = split_manifest or settings.split_manifest_path
    if not manifest_path.exists():
        raise typer.BadParameter(
            "Create the scene split manifest with build-splits before calibration"
        )
    manifest = load_scene_split_manifest(manifest_path)
    images = sorted(
        path
        for path in srgb_root.rglob("*")
        if path.is_file() and path.name.upper().startswith("GT_SRGB_")
        and path.suffix.upper() == ".PNG"
        and manifest["assignments"].get("_".join(path.parent.name.split("_")[:2]))
        == "train"
    )
    if max_samples is not None:
        images = images[:max_samples]
    if not images:
        raise typer.BadParameter(
            "No train-split GT_SRGB_*.PNG images were found under --srgb_root"
        )
    payload = ImageEvaluator.calibrate(images, output)
    typer.echo(
        f"Calibrated quality evaluator with {payload['sample_count']} GT sRGB images: {output}"
    )


@app.command("build-pseudo-labels")
def build_pseudo_labels_command(
    data_root: Annotated[
        Path | None,
        typer.Option("--data_root", help="SIDD Small Raw Only root."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Qwen SFT pseudo-label JSONL."),
    ] = Path("outputs/pseudo_labels_sft.jsonl"),
    audit_output: Annotated[
        Path,
        typer.Option("--audit_output", help="Full candidate audit JSONL."),
    ] = Path("outputs/pseudo_labels_audit.jsonl"),
    split_manifest: Annotated[
        Path | None,
        typer.Option("--split_manifest", help="Scene split manifest."),
    ] = None,
    quality_calibration: Annotated[
        Path | None,
        typer.Option("--quality_calibration", help="Optional quality calibration JSON."),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max_samples", help="Optional sample limit."),
    ] = None,
    exposure_offsets: Annotated[
        str,
        typer.Option(
            "--exposure_offsets",
            help="Comma-separated synthetic source EV offsets.",
        ),
    ] = "-1.0,0.0,1.0",
) -> None:
    settings = get_settings()
    root = data_root or settings.sidd_data_root
    reader = SIDDReader(root, preview_dir=output.parent / "previews")
    manifest_path = split_manifest or settings.split_manifest_path
    manifest = ensure_scene_split_manifest(
        reader.scene_ids(),
        manifest_path,
        seed=42,
        physical_scene_ids=reader.physical_scene_ids(),
    )
    calibration_path = (
        quality_calibration
        or settings.quality_calibration_path
        or output.parent / "quality_calibration.json"
    )
    evaluator = ImageEvaluator(
        calibration_path=calibration_path if calibration_path.exists() else None
    )
    policy = Policy()
    optimizer = LocalSearchOptimizer(
        evaluator=evaluator,
        policy=policy,
        candidate_dir=settings.artifacts_dir / "pseudo_label_search",
    )
    builder = SearchPseudoLabelBuilder(
        optimizer=optimizer,
        evaluator=evaluator,
        policy=policy,
        preview_dir=output.parent / "previews",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    writer = TrainingExampleWriter(output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text("", encoding="utf-8")
    processed = 0
    variant_count = 0
    written = 0
    try:
        offsets = tuple(float(token.strip()) for token in exposure_offsets.split(","))
    except ValueError as exc:
        raise typer.BadParameter("--exposure_offsets must contain numbers") from exc
    if not offsets or any(abs(offset) > 3.0 for offset in offsets):
        raise typer.BadParameter("Provide EV offsets in the range -3..3")
    with audit_output.open("a", encoding="utf-8") as audit_file:
        for sample in reader.iter_samples(max_samples=max_samples):
            split = split_for_scene(
                manifest,
                sample.scene_id,
                sample.physical_scene_id,
            )
            variants = builder.build_variants(
                sample,
                dataset_split=split,
                exposure_offsets_ev=offsets,
            )
            for example, audit in variants:
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                if example is not None:
                    written += writer.write_examples([example])
                variant_count += 1
            processed += 1
    typer.echo(
        f"Processed {processed} scenes/{variant_count} exposure states; wrote "
        f"{written} deduplicated pseudo labels "
        f"to {output}; audit: {audit_output}"
    )


@app.command("evaluate-actions")
def evaluate_actions_command(
    predictions: Annotated[
        Path,
        typer.Option("--predictions", help="Parsed ExposurePrediction JSONL."),
    ],
    targets: Annotated[
        Path,
        typer.Option("--targets", help="Pseudo-label SFT JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Metrics JSON path."),
    ] = Path("outputs/action_metrics.json"),
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to evaluate."),
    ] = "test",
) -> None:
    metrics = evaluate_action_predictions(
        predictions_path=predictions,
        targets_path=targets,
        split=split,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote Action evaluation metrics to {output}")


@app.command("evaluate-agent")
def evaluate_agent_command(
    runs: Annotated[
        Path,
        typer.Option("--runs", help="AgentResult JSONL from loop mode."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Agent metrics JSON path."),
    ] = Path("outputs/agent_metrics.json"),
    split: Annotated[
        str | None,
        typer.Option("--split", help="Optional dataset split filter."),
    ] = None,
) -> None:
    metrics = evaluate_agent_runs(runs_path=runs, split=split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote Agent evaluation metrics to {output}")


@app.command("train-qwen-vl")
def train_qwen_vl_command(
    train_jsonl: Annotated[
        Path,
        typer.Option("--train_jsonl", help="Input SFT JSONL from loop runs."),
    ] = Path("outputs/training_sft.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option("--output_dir", help="Checkpoint output directory."),
    ] = Path("checkpoints/qwen_vl_exposure_lora"),
    model_id: Annotated[
        str,
        typer.Option("--model_id", help="HuggingFace model id or local model path."),
    ] = "Qwen/Qwen2.5-VL-3B-Instruct",
    image_root: Annotated[
        Path,
        typer.Option("--image_root", help="Root for relative image paths in JSONL."),
    ] = Path("."),
    train_split: Annotated[
        str,
        typer.Option("--train_split", help="Only train on this scene split."),
    ] = "train",
    eval_jsonl: Annotated[
        Path | None,
        typer.Option("--eval_jsonl", help="Optional validation SFT JSONL."),
    ] = None,
    eval_split: Annotated[
        str,
        typer.Option("--eval_split", help="Validation split name."),
    ] = "validation",
    model_family: Annotated[
        str,
        typer.Option("--model_family", help="qwen2_5_vl or auto_image_text."),
    ] = "qwen2_5_vl",
    num_train_epochs: Annotated[
        float,
        typer.Option("--num_train_epochs", help="Number of training epochs."),
    ] = 3.0,
    max_steps: Annotated[
        int,
        typer.Option("--max_steps", help="Override total train steps; -1 disables."),
    ] = -1,
    learning_rate: Annotated[
        float,
        typer.Option("--learning_rate", help="Training learning rate."),
    ] = 2e-4,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option("--gradient_accumulation_steps", help="Effective batch accumulation."),
    ] = 8,
    max_length: Annotated[
        int,
        typer.Option("--max_length", help="Max token length."),
    ] = 4096,
    min_pixels: Annotated[
        int,
        typer.Option("--min_pixels", help="Minimum pixels used by the VLM image processor."),
    ] = 256 * 28 * 28,
    max_pixels: Annotated[
        int,
        typer.Option("--max_pixels", help="Maximum pixels used by the VLM image processor."),
    ] = 512 * 28 * 28,
    no_lora: Annotated[
        bool,
        typer.Option("--no_lora", help="Full fine-tuning instead of LoRA."),
    ] = False,
    no_bf16: Annotated[
        bool,
        typer.Option("--no_bf16", help="Disable bf16 training."),
    ] = False,
    fp16: Annotated[
        bool,
        typer.Option("--fp16", help="Enable fp16 training."),
    ] = False,
) -> None:
    from exposure_agent.train import QwenVLTrainingConfig, train_qwen_vl

    if model_family not in {"qwen2_5_vl", "auto_image_text"}:
        raise typer.BadParameter("--model_family must be qwen2_5_vl or auto_image_text")
    train_qwen_vl(
        QwenVLTrainingConfig(
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            model_id=model_id,
            image_root=image_root,
            train_split=train_split,
            eval_jsonl=eval_jsonl,
            eval_split=eval_split,
            model_family=model_family,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_lora=not no_lora,
            bf16=not no_bf16,
            fp16=fp16,
        )
    )


def _metadata_from_cli_values(
    *,
    reader: MetadataReader,
    image: Path,
    iso: int | None,
    shutter: float | None,
    ev: float | None,
    aperture: float | None,
):
    missing = [
        name
        for name, value in {
            "--iso": iso,
            "--shutter": shutter,
        }.items()
        if value is None
    ]
    if missing:
        raise typer.BadParameter(
            "Provide --metadata-json or the required CLI exposure parameters: "
            + ", ".join(missing)
        )
    return reader.from_values(
        iso=iso,
        shutter_speed_s=shutter,
        ev=ev,
        aperture=aperture,
        image_id=image.stem,
    )


def _build_agent(
    *,
    mode: str,
    memory_path: Path,
    use_rag: bool,
    use_local_search: bool,
) -> ExposureAgent:
    settings = get_settings()
    evaluator = _build_evaluator(settings)
    policy = Policy()
    optimizer = (
        LocalSearchOptimizer(
            evaluator=evaluator,
            policy=policy,
            candidate_dir=settings.artifacts_dir / "local_search",
        )
        if mode == "loop" and use_local_search
        else NoOpOptimizer()
    )
    memory = (
        JsonlMemory(memory_path, evaluator=evaluator, min_quality_gain=0.02)
        if mode == "loop" and use_rag
        else NoOpMemory()
    )
    return ExposureAgent(
        vlm=build_vlm_client(settings, evaluator=evaluator),
        memory=memory,
        optimizer=optimizer,
        evaluator=evaluator,
        policy=policy,
        artifacts_dir=settings.artifacts_dir,
    )


def _build_evaluator(settings) -> ImageEvaluator:
    calibration = settings.quality_calibration_path
    return ImageEvaluator(
        calibration_path=calibration if calibration is not None and calibration.exists() else None
    )


def prediction_to_row(prediction: ExposurePrediction) -> dict:
    return prediction.model_dump()

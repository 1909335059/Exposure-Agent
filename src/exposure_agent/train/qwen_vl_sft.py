from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenVLTrainingConfig:
    train_jsonl: Path
    output_dir: Path
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_root: Path = Path(".")
    model_family: str = "qwen2_5_vl"
    num_train_epochs: float = 3.0
    train_split: str = "train"
    eval_jsonl: Path | None = None
    eval_split: str = "validation"
    max_steps: int = -1
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    max_length: int = 4096
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    report_to: str = "none"


class ExposureSFTDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        image_root: str | Path = ".",
        split: str | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        rows = _read_jsonl(self.jsonl_path)
        if split is not None:
            rows = [
                row
                for row in rows
                if row.get("dataset_split", "train") == split
            ]
        self.samples = [
            normalize_messages_for_qwen_vl(row, image_root=self.image_root)
            for row in rows
        ]
        if not self.samples:
            raise ValueError(f"No valid SFT samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return self.samples[index]


class QwenVLSFTCollator:
    def __init__(self, *, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[list[dict[str, Any]]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "This collator intentionally supports batch_size=1. "
                "Use gradient_accumulation_steps for larger effective batches."
            )
        messages = features[0]
        prompt_messages = messages[:-1]
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = _process_vision_info(messages)
        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[inputs["input_ids"] == pad_token_id] = -100
        if not (labels != -100).any().item():
            raise ValueError(
                "The assistant target was fully truncated. Increase max_length or "
                "reduce the serialized feature payload before training."
            )
        inputs["labels"] = labels
        return inputs


def train_qwen_vl(config: QwenVLTrainingConfig) -> None:
    torch, transformers = _load_transformers_stack()
    if config.min_pixels <= 0 or config.max_pixels < config.min_pixels:
        raise ValueError("Require 0 < min_pixels <= max_pixels")
    processor = transformers.AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model_cls = _resolve_model_class(transformers, config.model_family)
    dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32
    model = model_cls.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if config.use_lora:
        model = _apply_lora(model, config)
    dataset = ExposureSFTDataset(
        config.train_jsonl,
        image_root=config.image_root,
        split=config.train_split,
    )
    eval_dataset = (
        ExposureSFTDataset(
            config.eval_jsonl,
            image_root=config.image_root,
            split=config.eval_split,
        )
        if config.eval_jsonl is not None
        else None
    )
    collator = QwenVLSFTCollator(processor=processor, max_length=config.max_length)
    training_args = transformers.TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "steps",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))


def normalize_messages_for_qwen_vl(
    row: dict[str, Any],
    *,
    image_root: str | Path = ".",
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row must contain a messages list")
    root = Path(image_root)
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a JSON object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        normalized.append(
            {
                "role": role,
                "content": _normalize_content(content, image_root=root),
            }
        )
    return normalized


def _normalize_content(content: Any, *, image_root: Path) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be a string or a list")
    normalized: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Multimodal content items must be JSON objects")
        if "image" in item:
            image_path = Path(str(item["image"]))
            if not image_path.is_absolute():
                image_path = image_root / image_path
            normalized.append({"type": "image", "image": str(image_path)})
        elif "text" in item:
            normalized.append({"type": "text", "text": str(item["text"])})
        elif item.get("type") == "image" and "image" in item:
            image_path = Path(str(item["image"]))
            if not image_path.is_absolute():
                image_path = image_root / image_path
            normalized.append({"type": "image", "image": str(image_path)})
        elif item.get("type") == "text" and "text" in item:
            normalized.append({"type": "text", "text": str(item["text"])})
        else:
            raise ValueError(f"Unsupported multimodal content item: {item}")
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _process_vision_info(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "qwen-vl-utils is required for Qwen-VL training. "
            "Install training dependencies with: pip install -e '.[train]'"
        ) from exc
    return process_vision_info(messages)


def _load_transformers_stack() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required for VLM training. "
            "Install training dependencies with: pip install -e '.[train]'"
        ) from exc
    return torch, transformers


def _resolve_model_class(transformers: Any, model_family: str) -> Any:
    if model_family == "qwen2_5_vl":
        try:
            return transformers.Qwen2_5_VLForConditionalGeneration
        except AttributeError as exc:
            raise RuntimeError(
                "Your transformers version does not expose "
                "Qwen2_5_VLForConditionalGeneration. Upgrade transformers "
                "or use --model_family auto_image_text."
            ) from exc
    if model_family == "auto_image_text":
        try:
            return transformers.AutoModelForImageTextToText
        except AttributeError as exc:
            raise RuntimeError(
                "Your transformers version does not expose "
                "AutoModelForImageTextToText."
            ) from exc
    raise ValueError(f"Unsupported model_family: {model_family}")


def _apply_lora(model: Any, config: QwenVLTrainingConfig) -> Any:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA training. "
            "Install training dependencies with: pip install -e '.[train]'"
        ) from exc
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA/SFT train an open-source Qwen-VL model.")
    parser.add_argument("--train_jsonl", type=Path, default=Path("outputs/training_sft.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/qwen_vl_exposure_lora"))
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_root", type=Path, default=Path("."))
    parser.add_argument("--model_family", default="qwen2_5_vl", choices=["qwen2_5_vl", "auto_image_text"])
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_jsonl", type=Path)
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--report_to", default="none")
    return parser


def config_from_args(args: argparse.Namespace) -> QwenVLTrainingConfig:
    return QwenVLTrainingConfig(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        model_id=args.model_id,
        image_root=args.image_root,
        model_family=args.model_family,
        num_train_epochs=args.num_train_epochs,
        train_split=args.train_split,
        eval_jsonl=args.eval_jsonl,
        eval_split=args.eval_split,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        bf16=not args.no_bf16,
        fp16=args.fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        trust_remote_code=not args.no_trust_remote_code,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        report_to=args.report_to,
    )


def main() -> None:
    parser = build_arg_parser()
    train_qwen_vl(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

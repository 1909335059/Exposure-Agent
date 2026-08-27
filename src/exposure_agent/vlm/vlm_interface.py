from __future__ import annotations

import base64
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from exposure_agent.config import Settings
from exposure_agent.evaluator import ImageEvaluator
from exposure_agent.models import (
    ExposureAction,
    ExposureMetadata,
    ImageFeatureBundle,
    PreviousRoundFeedback,
    VLMDecision,
)
from exposure_agent.vlm.parser import VLMParseError, parse_vlm_decision
from exposure_agent.vlm.prompt import (
    build_experience_integration_prompt,
    build_initial_exposure_prompt,
)


class VLMInterface(ABC):
    @abstractmethod
    def propose_initial(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        raise NotImplementedError

    @abstractmethod
    def integrate_experience(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        initial_decision: VLMDecision,
        memory_context: dict | None,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        raise NotImplementedError


class MockVLMClient(VLMInterface):
    """Deterministic stand-in that preserves the two-pass VLM call contract."""

    def __init__(self, evaluator: ImageEvaluator | None = None) -> None:
        self.evaluator = evaluator or ImageEvaluator()

    def propose_initial(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        report = (
            feedback.objective_quality
            if feedback is not None
            else fixed_features.objective_quality
        )
        reference = feedback.result_metadata if feedback is not None else metadata
        action = _heuristic_target(report=report, metadata=reference)
        return VLMDecision(
            quality=report.quality,
            action=action,
            continue_adjustment=not report.acceptable,
            reason="mock_initial_recommendation",
        )

    def integrate_experience(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        initial_decision: VLMDecision,
        memory_context: dict | None,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        experience_action = _best_memory_action(memory_context)
        action = (
            _blend_targets(initial_decision.action, experience_action)
            if experience_action is not None
            else initial_decision.action
        )
        return VLMDecision(
            quality=initial_decision.quality,
            action=action,
            continue_adjustment=initial_decision.continue_adjustment,
            reason=(
                "mock_integrated_retrieved_experience"
                if experience_action is not None
                else "mock_integrated_without_retrieval"
            ),
        )


class _PromptedVLMClient(VLMInterface, ABC):
    def propose_initial(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        prompt = build_initial_exposure_prompt(
            metadata=metadata,
            fixed_features=fixed_features,
            feedback=feedback,
        )
        return self._run_prompt(
            prompt=prompt,
            image_paths=_image_paths(original_image_path, feedback),
        )

    def integrate_experience(
        self,
        *,
        original_image_path: str | Path,
        metadata: ExposureMetadata,
        fixed_features: ImageFeatureBundle,
        initial_decision: VLMDecision,
        memory_context: dict | None,
        feedback: PreviousRoundFeedback | None = None,
    ) -> VLMDecision:
        prompt = build_experience_integration_prompt(
            metadata=metadata,
            fixed_features=fixed_features,
            initial_decision=initial_decision,
            memory_context=memory_context,
            feedback=feedback,
        )
        return self._run_prompt(
            prompt=prompt,
            image_paths=_image_paths(original_image_path, feedback),
        )

    @abstractmethod
    def _run_prompt(self, *, prompt: str, image_paths: list[Path]) -> VLMDecision:
        raise NotImplementedError


class OllamaVLMClient(_PromptedVLMClient):
    def __init__(
        self,
        *,
        url: str,
        model: str,
        evaluator: ImageEvaluator | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if not model:
            raise ValueError("EXPOSURE_OLLAMA_MODEL is required when backend=ollama")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.evaluator = evaluator or ImageEvaluator()

    def _run_prompt(self, *, prompt: str, image_paths: list[Path]) -> VLMDecision:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [
                base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths
            ],
            "stream": False,
            "format": "json",
        }
        response = httpx.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "response" not in body:
            raise ValueError("Ollama response did not contain a response field")
        return parse_vlm_decision(body["response"])


class DashScopeQwenVLMClient(_PromptedVLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        evaluator: ImageEvaluator | None = None,
        enable_thinking: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY or EXPOSURE_DASHSCOPE_API_KEY is required")
        if not model:
            raise ValueError("EXPOSURE_DASHSCOPE_MODEL is required when backend=dashscope")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.enable_thinking = enable_thinking
        self.evaluator = evaluator or ImageEvaluator()

    def _run_prompt(self, *, prompt: str, image_paths: list[Path]) -> VLMDecision:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as exc:
            raise RuntimeError(
                "dashscope is required for EXPOSURE_BACKEND=dashscope. "
                "Install it with: python -m pip install dashscope"
            ) from exc

        dashscope.base_http_api_url = self.base_url
        user_content = [{"image": path.resolve().as_uri()} for path in image_paths]
        user_content.append({"text": prompt})
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "text": (
                                "You are an exposure-control VLM. "
                                "Return one valid JSON object only."
                            )
                        }
                    ],
                },
                {"role": "user", "content": user_content},
            ],
            result_format="message",
            enable_thinking=self.enable_thinking,
        )
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            code = getattr(response, "code", "unknown")
            message = getattr(response, "message", "unknown error")
            raise RuntimeError(f"DashScope request failed: {status_code} {code}: {message}")
        return parse_vlm_decision(_extract_dashscope_content(response))


class LocalQwenVLVLMClient(_PromptedVLMClient):
    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | Path | None = None,
        model_family: str = "qwen2_5_vl",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        evaluator: ImageEvaluator | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("EXPOSURE_LOCAL_QWEN_VL_MODEL_ID is required")
        self.model_id = model_id
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.model_family = model_family
        self.max_new_tokens = max_new_tokens
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Require 0 < min_pixels <= max_pixels")
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.evaluator = evaluator or ImageEvaluator()
        self.processor, self.model = self._load_model()

    def _run_prompt(self, *, prompt: str, image_paths: list[Path]) -> VLMDecision:
        user_content = [
            {"type": "image", "image": str(path.resolve())} for path in image_paths
        ]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "system",
                "content": "You are an exposure-control VLM. Return one valid JSON object only.",
            },
            {"role": "user", "content": user_content},
        ]
        current_messages = messages
        last_error: VLMParseError | None = None
        for _ in range(3):
            output = self._generate(current_messages)
            try:
                return parse_vlm_decision(output)
            except VLMParseError as exc:
                last_error = exc
                current_messages = [
                    *current_messages,
                    {"role": "assistant", "content": output},
                    {
                        "role": "user",
                        "content": (
                            "Your response failed strict JSON validation. Return exactly one "
                            "complete JSON object with all required fields: quality must contain "
                            "brightness, noise, motion_blur, highlight, shadow, and "
                            "overall_quality numbers from 0 to 1; action must contain absolute "
                            "ISO and Shutter; continue must be boolean; reason is optional. "
                            "Do not omit quality, do not output EV, and do not add text outside "
                            f"JSON. Validation error: {exc}"
                        ),
                    },
                ]
        raise last_error or VLMParseError("Local Qwen failed strict JSON validation")

    def _load_model(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for local_qwen_vl. "
                "Install training dependencies with: pip install -e '.[train]'"
            ) from exc
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        model_cls = _resolve_local_qwen_model_class(transformers, self.model_family)
        model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required to load a LoRA adapter. "
                    "Install training dependencies with: pip install -e '.[train]'"
                ) from exc
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        model.eval()
        return processor, model

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "qwen-vl-utils is required for local_qwen_vl. "
                "Install training dependencies with: pip install -e '.[train]'"
            ) from exc
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _image_paths(
    original_image_path: str | Path,
    feedback: PreviousRoundFeedback | None,
) -> list[Path]:
    paths = [Path(original_image_path)]
    if feedback is not None:
        paths.append(Path(feedback.result_image_path))
    return paths


def _best_memory_action(memory_context: dict | None) -> ExposureAction | None:
    if not isinstance(memory_context, dict):
        return None
    experience = memory_context.get("best_experience")
    if not isinstance(experience, dict):
        examples = memory_context.get("examples")
        experience = examples[0] if isinstance(examples, list) and examples else None
    if not isinstance(experience, dict):
        return None
    payload = experience.get("final_action")
    if not isinstance(payload, dict):
        return None
    try:
        return ExposureAction.model_validate(payload)
    except ValueError:
        return None


def _blend_targets(
    initial: ExposureAction,
    experience: ExposureAction,
    *,
    initial_weight: float = 0.70,
) -> ExposureAction:
    memory_weight = 1.0 - initial_weight
    shutter = math.exp(
        initial_weight * math.log(initial.target_shutter_speed_s)
        + memory_weight * math.log(experience.target_shutter_speed_s)
    )
    return ExposureAction(
        target_iso=round(
            initial_weight * initial.target_iso + memory_weight * experience.target_iso
        ),
        target_shutter_speed_s=shutter,
    )


def _heuristic_target(*, report: Any, metadata: ExposureMetadata) -> ExposureAction:
    metrics = report.quality
    iso = metadata.iso
    shutter = metadata.shutter_speed_s
    if metrics.highlight > 0.03:
        shutter *= 0.6
    elif metrics.shadow > 0.30 or report.midtone_ratio < 0.55:
        if metrics.motion_blur > 0.45:
            iso *= 2
        else:
            shutter *= 1.5
    elif metrics.noise > 0.45:
        iso *= 0.5
        shutter *= 2.0
    elif metrics.motion_blur > 0.55 and report.sharpness_confidence > 0.5:
        iso *= 2
        shutter *= 0.5
    return ExposureAction(
        target_iso=max(1, round(iso)),
        target_shutter_speed_s=shutter,
    )


def _extract_dashscope_content(response: Any) -> str:
    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None)
    if not choices:
        raise ValueError("DashScope response did not contain output choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "\n".join(text_parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    raise ValueError("DashScope response message content was empty or unsupported")


def _resolve_local_qwen_model_class(transformers: Any, model_family: str) -> Any:
    if model_family == "qwen2_5_vl":
        try:
            return transformers.Qwen2_5_VLForConditionalGeneration
        except AttributeError as exc:
            raise RuntimeError(
                "Your transformers version does not expose "
                "Qwen2_5_VLForConditionalGeneration. Upgrade transformers "
                "or set EXPOSURE_LOCAL_QWEN_VL_MODEL_FAMILY=auto_image_text."
            ) from exc
    if model_family == "auto_image_text":
        try:
            return transformers.AutoModelForImageTextToText
        except AttributeError as exc:
            raise RuntimeError(
                "Your transformers version does not expose AutoModelForImageTextToText."
            ) from exc
    raise ValueError(f"Unsupported local_qwen_vl model family: {model_family}")


def build_vlm_client(
    settings: Settings,
    *,
    evaluator: ImageEvaluator | None = None,
) -> VLMInterface:
    if settings.backend == "mock":
        return MockVLMClient(evaluator=evaluator)
    if settings.backend == "ollama":
        return OllamaVLMClient(
            url=settings.ollama_url,
            model=settings.ollama_model,
            evaluator=evaluator,
        )
    if settings.backend == "dashscope":
        return DashScopeQwenVLMClient(
            api_key=settings.dashscope_api_key,
            model=settings.dashscope_model,
            base_url=settings.dashscope_base_url,
            evaluator=evaluator,
            enable_thinking=settings.dashscope_enable_thinking,
        )
    if settings.backend == "local_qwen_vl":
        return LocalQwenVLVLMClient(
            model_id=settings.local_qwen_vl_model_id,
            adapter_path=settings.local_qwen_vl_adapter_path,
            model_family=settings.local_qwen_vl_model_family,
            max_new_tokens=settings.local_qwen_vl_max_new_tokens,
            min_pixels=settings.local_qwen_vl_min_pixels,
            max_pixels=settings.local_qwen_vl_max_pixels,
            evaluator=evaluator,
        )
    raise ValueError(f"Unsupported VLM backend: {settings.backend}")

from __future__ import annotations

import json

from pydantic import ValidationError

from exposure_agent.models import VLMDecision


class VLMParseError(ValueError):
    pass


def parse_vlm_decision(raw_response: str | bytes | dict) -> VLMDecision:
    if isinstance(raw_response, dict):
        payload = raw_response
    else:
        try:
            text = raw_response.decode("utf-8") if isinstance(raw_response, bytes) else raw_response
            text = _strip_json_fence(text)
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VLMParseError("VLM response must be a JSON object") from exc

    if not isinstance(payload, dict):
        raise VLMParseError("VLM response must be a JSON object")

    try:
        return VLMDecision.model_validate(payload)
    except ValidationError as exc:
        raise VLMParseError(str(exc)) from exc


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

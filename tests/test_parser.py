from __future__ import annotations

import pytest

from exposure_agent.models import ExposureMetadata
from exposure_agent.vlm import LocalQwenVLVLMClient
from exposure_agent.vlm.parser import VLMParseError, parse_vlm_decision

from tests.conftest import feature_bundle


VALID_JSON = """
{
  "quality": {
    "brightness": 0.42,
    "noise": 0.18,
    "motion_blur": 0.12,
    "highlight": 0.08,
    "shadow": 0.31,
    "overall_quality": 0.73
  },
  "action": {"ISO": 400, "Shutter": 0.025},
  "continue": true,
  "reason": "optional"
}
"""


def test_parse_vlm_decision_accepts_absolute_target_json() -> None:
    decision = parse_vlm_decision(VALID_JSON)

    assert decision.action.target_iso == 400
    assert decision.action.target_shutter_speed_s == 0.025
    assert decision.continue_adjustment is True


def test_parse_vlm_decision_accepts_json_markdown_fence() -> None:
    decision = parse_vlm_decision(f"```json\n{VALID_JSON}\n```")
    assert decision.action.target_iso == 400


@pytest.mark.parametrize(
    "raw",
    [
        "图片偏暗，建议提高ISO。",
        '{"quality": {"brightness": 0.5}}',
        (
            '{"quality":{"brightness":0.5,"noise":0.1,"motion_blur":0.1,'
            '"highlight":0,"shadow":0,"overall_quality":0.8},'
            '"action":{"ISO":100,"EV":0.3,"Shutter":0.02},"continue":true}'
        ),
    ],
)
def test_parse_vlm_decision_rejects_invalid_or_legacy_responses(raw: str) -> None:
    with pytest.raises(VLMParseError):
        parse_vlm_decision(raw)


def test_local_qwen_retries_once_after_schema_validation_failure(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color=(64, 64, 64)).save(image_path)
    client = object.__new__(LocalQwenVLVLMClient)
    outputs = iter(
        [
            '{"action":{"ISO":100,"Shutter":0.02},"continue":false}',
            VALID_JSON,
        ]
    )
    calls = []

    def fake_generate(messages):
        calls.append(messages)
        return next(outputs)

    monkeypatch.setattr(client, "_generate", fake_generate)
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60)

    decision = client.propose_initial(
        original_image_path=image_path,
        metadata=metadata,
        fixed_features=feature_bundle(),
    )

    assert len(calls) == 2
    assert decision.action.target_iso == 400
    assert "failed strict JSON validation" in calls[1][-1]["content"]

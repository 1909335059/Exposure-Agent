from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from exposure_agent.config import Settings
from exposure_agent.models import ExposureMetadata
from exposure_agent.vlm import DashScopeQwenVLMClient, build_vlm_client
from exposure_agent.vlm.vlm_interface import _extract_dashscope_content

from tests.conftest import feature_bundle


def test_extract_dashscope_content_from_string_message() -> None:
    response = SimpleNamespace(
        output=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
    )

    assert _extract_dashscope_content(response) == '{"ok": true}'


def test_extract_dashscope_content_from_multimodal_message() -> None:
    response = SimpleNamespace(
        output=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=[{"text": '{"ok": true}'}])
                )
            ]
        )
    )

    assert _extract_dashscope_content(response) == '{"ok": true}'


def test_dashscope_client_parses_vlm_decision(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_path)
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"quality":{"brightness":0.5,"noise":0.1,'
                                '"motion_blur":0.2,"highlight":0.0,'
                                '"shadow":0.0,"overall_quality":0.8},'
                                '"action":{"ISO":100,"Shutter":0.0166666667},'
                                '"continue":false,"reason":"ok"}'
                            )
                        )
                    )
                ]
            ),
        )

    import dashscope

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", fake_call)
    client = DashScopeQwenVLMClient(
        api_key="test-key",
        model="qwen-test",
        base_url="https://example.com/api/v1",
    )

    decision = client.propose_initial(
        original_image_path=image_path,
        metadata=ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=5.9),
        fixed_features=feature_bundle(),
    )

    assert decision.continue_adjustment is False
    assert decision.action.target_shutter_speed_s > 0
    assert captured["model"] == "qwen-test"
    user_content = captured["messages"][1]["content"]
    assert user_content[0]["image"].startswith("file://")


def test_build_vlm_client_supports_dashscope(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    settings = Settings(backend="dashscope")

    client = build_vlm_client(settings)

    assert isinstance(client, DashScopeQwenVLMClient)

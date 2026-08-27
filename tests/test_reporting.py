from __future__ import annotations

from exposure_agent.models import (
    AgentIteration,
    AgentResult,
    ExposureAction,
    ExposureMetadata,
)
from exposure_agent.reporting import StructuredReportWriter, build_structured_report

from tests.conftest import feature_bundle, quality_report, vlm_decision


def _result(tmp_path) -> AgentResult:
    metadata = ExposureMetadata(image_id="sample", iso=100, shutter_speed_s=1 / 60)
    updated = ExposureMetadata(image_id="sample", iso=200, shutter_speed_s=1 / 30, ev=3)
    before = quality_report(0.4, acceptable=False, shadow=0.6)
    after = quality_report(0.75, acceptable=True)
    features = feature_bundle(before)
    initial = vlm_decision(metadata, report=before, target_iso=200, target_shutter=1 / 40)
    integrated = vlm_decision(
        metadata,
        report=before,
        target_iso=200,
        target_shutter=1 / 30,
    )
    final_action = ExposureAction(target_iso=200, target_shutter_speed_s=1 / 30)
    return AgentResult(
        run_id="sample",
        scene_id="sample",
        original_image_path=tmp_path / "input.png",
        initial_metadata=metadata,
        fixed_features=features,
        iterations=[
            AgentIteration(
                index=1,
                original_image_path=tmp_path / "input.png",
                initial_metadata=metadata,
                fixed_features=features,
                initial_vlm_decision=initial,
                memory_context={
                    "best_experience": {
                        "scene_id": "other",
                        "retrieval_score": 0.1,
                        "final_action": {"ISO": 200, "Shutter": 1 / 30},
                    }
                },
                integrated_vlm_decision=integrated,
                semi_final_action=integrated.action,
                final_action=final_action,
                final_metadata=updated,
                output_image_path=tmp_path / "output.png",
                objective_quality_before=before,
                objective_quality_after=after,
                quality_gain_from_original=0.35,
                satisfactory=True,
                optimizer_trace={
                    "enabled": True,
                    "search_type": "two_stage_around_integrated_vlm_target",
                    "candidate_count": 18,
                    "best_quality": 0.75,
                },
                stop_reason="quality_satisfactory",
            )
        ],
        final_image_path=tmp_path / "output.png",
        final_metadata=updated,
        final_quality=after.quality,
        final_objective_quality=after,
        stop_reason="quality_satisfactory",
    )


def test_structured_report_follows_agreed_pipeline_sections(tmp_path) -> None:
    report = build_structured_report(_result(tmp_path))

    first_round = report["rounds"][0]
    assert report["pipeline"] == "two_pass_vlm_rag_local_search_three_round_v1"
    assert "fixed_features" in report["original_input"]
    assert "vlm_first_recommendation" in first_round
    assert first_round["rag_retrieval"]["retrieved"] is True
    assert "vlm_experience_integrated_recommendation" in first_round
    assert first_round["local_search"]["candidate_count"] == 18
    assert first_round["output"]["satisfactory"] is True


def test_structured_writer_also_writes_chinese_markdown_and_images(tmp_path) -> None:
    output = tmp_path / "report.json"
    writer = StructuredReportWriter(output)
    writer.add_result(_result(tmp_path))
    writer.close()

    markdown = output.with_suffix(".md")
    text = markdown.read_text(encoding="utf-8")
    assert output.exists()
    assert "原始输入" in text
    assert "第一次 VLM 建议" in text
    assert "第二次 VLM 综合建议" in text
    assert "最终最佳图像" in text

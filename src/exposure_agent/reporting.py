from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exposure_agent.models import AgentResult


class StructuredReportWriter:
    def __init__(
        self,
        output_path: str | Path,
        *,
        markdown_output_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.markdown_output_path = (
            Path(markdown_output_path)
            if markdown_output_path is not None
            else self.output_path.with_suffix(".md")
        )
        self.results: list[AgentResult] = []

    def add_result(self, result: AgentResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        reports = [build_structured_report(result) for result in self.results]
        payload: dict | list = reports[0] if len(reports) == 1 else reports
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.markdown_output_path.write_text(
            build_chinese_markdown_report(self.results),
            encoding="utf-8",
        )


def build_structured_report(result: AgentResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "scene_id": result.scene_id,
        "dataset_split": result.dataset_split,
        "pipeline": "two_pass_vlm_rag_local_search_three_round_v1",
        "original_input": {
            "image_path": str(result.original_image_path),
            "metadata": result.initial_metadata.model_dump(),
            "fixed_features": result.fixed_features.model_dump(),
        },
        "rounds": [_round_report(iteration) for iteration in result.iterations],
        "final_result": {
            "best_image_path": str(result.final_image_path),
            "best_metadata": result.final_metadata.model_dump(),
            "best_objective_quality": result.final_objective_quality.model_dump(),
            "stop_reason": result.stop_reason,
            "round_count": len(result.iterations),
        },
    }


def build_chinese_markdown_report(results: list[AgentResult]) -> str:
    lines = ["# ExposureAgent 实验报告", ""]
    for result in results:
        lines.extend(
            [
                f"## 场景 {result.scene_id or result.run_id}",
                "",
                f"- 数据划分：`{result.dataset_split}`",
                f"- 最终停止原因：`{result.stop_reason}`",
                f"- 实际轮数：`{len(result.iterations)}`",
                "",
                "### 原始输入",
                "",
                f"![原始输入图像]({result.original_image_path})",
                "",
                f"- 初始参数：`{json.dumps(result.initial_metadata.model_dump(), ensure_ascii=False)}`",
                f"- 亮度直方图 bins：`{result.fixed_features.histogram_bins}`",
                "",
            ]
        )
        for iteration in result.iterations:
            lines.extend(
                [
                    f"### 第 {iteration.index} 轮",
                    "",
                    f"- 上一轮反馈：`{_feedback_text(iteration.feedback_input)}`",
                    f"- 第一次 VLM 建议：`{_decision_json(iteration.initial_vlm_decision)}`",
                    f"- RAG 检索：`{_rag_summary(iteration.memory_context)}`",
                    f"- 第二次 VLM 综合建议：`{_decision_json(iteration.integrated_vlm_decision)}`",
                    f"- 网格搜索最终目标：`{_action_json(iteration.final_action)}`",
                    f"- 最终参数：`{json.dumps(iteration.final_metadata.model_dump(), ensure_ascii=False)}`",
                    f"- 相对原图质量增益：`{iteration.quality_gain_from_original:.6f}`",
                    f"- 曝光是否满意：`{iteration.satisfactory}`",
                    "",
                    f"![第 {iteration.index} 轮输出图像]({iteration.output_image_path})",
                    "",
                ]
            )
        lines.extend(
            [
                "### 最终最佳结果",
                "",
                f"![最终最佳图像]({result.final_image_path})",
                "",
                f"- 最终参数：`{json.dumps(result.final_metadata.model_dump(), ensure_ascii=False)}`",
                f"- 最终客观质量：`{json.dumps(result.final_objective_quality.model_dump(), ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _round_report(iteration) -> dict[str, Any]:
    trace = iteration.optimizer_trace or {}
    return {
        "round": iteration.index,
        "previous_round_feedback": (
            iteration.feedback_input.model_dump(mode="json")
            if iteration.feedback_input is not None
            else None
        ),
        "vlm_first_recommendation": iteration.initial_vlm_decision.model_dump(
            mode="json", by_alias=True
        ),
        "rag_retrieval": _rag_report(iteration.memory_context),
        "vlm_experience_integrated_recommendation": (
            iteration.integrated_vlm_decision.model_dump(mode="json", by_alias=True)
        ),
        "local_search": {
            "enabled": bool(trace.get("enabled")),
            "search_type": trace.get("search_type"),
            "semi_final_action": iteration.semi_final_action.model_dump(by_alias=True),
            "candidate_count": trace.get("candidate_count", 0),
            "baseline_quality": trace.get("baseline_quality"),
            "best_action": iteration.final_action.model_dump(by_alias=True),
            "best_quality": trace.get("best_quality"),
            "best_gain": trace.get("best_gain"),
            "best_candidate": trace.get("best_candidate"),
            "candidates": trace.get("candidates", []),
        },
        "output": {
            "image_path": str(iteration.output_image_path),
            "metadata": iteration.final_metadata.model_dump(),
            "objective_quality": iteration.objective_quality_after.model_dump(),
            "quality_gain_from_original": iteration.quality_gain_from_original,
            "satisfactory": iteration.satisfactory,
            "stop_reason": iteration.stop_reason,
        },
    }


def _rag_report(memory_context: dict | None) -> dict[str, Any]:
    if not isinstance(memory_context, dict):
        return {"retrieved": False, "best_experience": None}
    return {
        "retrieved": memory_context.get("best_experience") is not None,
        "retrieval_type": memory_context.get("retrieval_type"),
        "distance_weights": memory_context.get("distance_weights"),
        "best_experience": memory_context.get("best_experience"),
    }


def _decision_json(decision) -> str:
    return json.dumps(decision.model_dump(by_alias=True), ensure_ascii=False)


def _action_json(action) -> str:
    return json.dumps(action.model_dump(by_alias=True), ensure_ascii=False)


def _feedback_text(feedback) -> str:
    if feedback is None:
        return "无，使用原始输入"
    return json.dumps(feedback.model_dump(mode="json"), ensure_ascii=False)


def _rag_summary(memory_context: dict | None) -> str:
    if not isinstance(memory_context, dict):
        return "未检索到经验"
    return json.dumps(memory_context.get("best_experience"), ensure_ascii=False)

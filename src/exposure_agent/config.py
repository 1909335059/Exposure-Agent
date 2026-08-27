from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXPOSURE_",
        env_file=".env",
        extra="ignore",
    )

    backend: Literal["mock", "ollama", "dashscope", "local_qwen_vl"] = "mock"
    run_mode: Literal["single", "loop"] = "single"
    artifacts_dir: Path = Path("artifacts")
    sidd_data_root: Path = Path(
        "/Users/sh1we1pen9/Coding/Datasets/SIDD/SIDD_Small_Raw_Only"
    )
    predictions_output: Path = Path("outputs/predictions.jsonl")
    memory_path: Path = Path("outputs/memory.jsonl")
    training_output: Path = Path("outputs/training_sft.jsonl")
    split_manifest_path: Path = Path("outputs/sidd_scene_splits.json")
    quality_calibration_path: Path | None = None
    max_iterations: int = Field(default=3, ge=1, le=3)
    enable_rag: bool = True
    enable_local_search: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    dashscope_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EXPOSURE_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY"),
    )
    dashscope_model: str = "qwen3.6-35b-a3b"
    dashscope_base_url: str = (
        "https://ws-ze8lcqgb15mb6heo.cn-beijing.maas.aliyuncs.com/api/v1"
    )
    dashscope_enable_thinking: bool = False
    local_qwen_vl_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_qwen_vl_adapter_path: Path | None = None
    local_qwen_vl_model_family: Literal["qwen2_5_vl", "auto_image_text"] = "qwen2_5_vl"
    local_qwen_vl_max_new_tokens: int = Field(default=512, ge=1)
    local_qwen_vl_min_pixels: int = Field(default=256 * 28 * 28, ge=1)
    local_qwen_vl_max_pixels: int = Field(default=512 * 28 * 28, ge=1)


def get_settings() -> Settings:
    return Settings()

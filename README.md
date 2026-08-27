# ExposureAgent

论文实验版：基于 VLM 的曝光参数调整 Agent。

当前代码支持两种实验模式：

- `single`：提取固定图像特征，依次运行第一次 VLM 建议和第二次 VLM 综合建议，输出绝对 ISO / shutter 目标。
- `loop`：运行正式论文 Pipeline。每轮执行第一次 VLM、RAG 检索、第二次 VLM 综合、局部搜索和模拟拍摄；不满意结果作为下一轮反馈，最多 3 轮。

注意：SIDD 没有同一场景不同曝光参数的真实重拍序列，所以 `loop` 模式里的下一帧图像来自 `ExposureSimulator` 的近似模拟，适合做 Agent 框架和论文方法验证，不等同于真实相机重拍。

## Pipeline

```text
原始相机图像 + 初始 ISO/快门
  -> 一次性提取固定视觉特征、亮度直方图和质量特征
  -> VLM 第一次建议
  -> RAG 按固定特征和第一次建议检索最佳历史经验
  -> VLM 第二次结合历史经验生成半最终目标
  -> 在半最终 ISO/快门附近进行两阶段局部搜索
  -> Simulator 生成本轮结果图
  -> Evaluator 判断是否满意
       -> 满意：输出并按增益写入 Memory
       -> 不满意：把本轮结果图、Action 和未达标项反馈给下一轮第一次 VLM
  -> 最多 3 轮，超限时返回全程客观质量最高的图像
```

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Environment

复制 `.env.example` 为 `.env`，然后按需修改：

```bash
EXPOSURE_BACKEND=mock
EXPOSURE_RUN_MODE=single
EXPOSURE_ARTIFACTS_DIR=artifacts
EXPOSURE_SIDD_DATA_ROOT=/Users/sh1we1pen9/Coding/Datasets/SIDD/SIDD_Small_Raw_Only
EXPOSURE_PREDICTIONS_OUTPUT=outputs/predictions.jsonl
EXPOSURE_MEMORY_PATH=outputs/memory.jsonl
EXPOSURE_TRAINING_OUTPUT=outputs/training_sft.jsonl
EXPOSURE_SPLIT_MANIFEST_PATH=outputs/sidd_scene_splits.json
EXPOSURE_QUALITY_CALIBRATION_PATH=outputs/quality_calibration.json
EXPOSURE_MAX_ITERATIONS=3
EXPOSURE_ENABLE_RAG=true
EXPOSURE_ENABLE_LOCAL_SEARCH=true
```

DashScope / 千问：

```bash
EXPOSURE_BACKEND=dashscope
DASHSCOPE_API_KEY=your_api_key
EXPOSURE_DASHSCOPE_MODEL=qwen3.6-35b-a3b
EXPOSURE_DASHSCOPE_BASE_URL=https://ws-ze8lcqgb15mb6heo.cn-beijing.maas.aliyuncs.com/api/v1
EXPOSURE_DASHSCOPE_ENABLE_THINKING=false
```

## Run Single-Step

SIDD 单步预测：

```bash
python main.py \
  --dataset sidd \
  --mode single \
  --max_samples 10
```

输出：

- `outputs/predictions.jsonl`：每行一个 `ExposurePrediction`
- `outputs/previews/*.png`：SIDD RAW 渲染出的 sRGB preview

单张图像：

```bash
python main.py run path/to/image.jpg --iso 400 --shutter 0.0167
```

## Run Closed Loop

完整论文 idea 的模拟闭环：

```bash
python main.py \
  --dataset sidd \
  --mode loop \
  --max_samples 10 \
  --max_iterations 3 \
  --output outputs/agent_runs.jsonl \
  --memory_path outputs/memory.jsonl \
  --training_output outputs/training_sft.jsonl
```

输出：

- `outputs/agent_runs.jsonl`：每轮固定特征、第一次 VLM 建议、RAG item、第二次 VLM 建议、搜索目标、反馈和结果图像。
- `outputs/memory.jsonl`：普通追加式 RAG 经验库，只保存训练场景中质量增益至少 `0.02` 的最佳策略。
- `outputs/training_sft.jsonl`：包含 `initial` 和 `integration` 两阶段的 Qwen-VL 微调样本。
- `outputs/sidd_scene_splits.json`：固定 seed=42、按 scene ID 划分的 train/validation/test manifest。
- `outputs/structured_report.json` 与同名 `.md`：按论文 Pipeline 保存每轮结构化结果和中英文可读图像报告。
- `artifacts/<run_id>/round_*.png`：每轮模拟得到的结果图像。
- `artifacts/local_search/.../*.png`：局部搜索候选图像。

如果只想跑闭环框架、不跑局部搜索：

```bash
python main.py --dataset sidd --mode loop --max_samples 10 --no_local_search
```

如果只想关掉 RAG：

```bash
python main.py --dataset sidd --mode loop --max_samples 10 --no_rag
```

RAG 根据固定视觉描述、32-bin 亮度直方图、质量向量、初始曝光参数和第一次 VLM 建议检索最相似成功经验。检索排除相同 scene/run，validation/test 只读。旧格式 Memory 会被忽略，正式实验前应重新生成；自进化、合并和淘汰机制不启用。

## Build Search Pseudo Labels

先生成固定场景划分：

```bash
python main.py build-splits \
  --data_root /path/to/SIDD_Small_Raw_Only \
  --output outputs/sidd_scene_splits.json
```

再用 train 划分中的配对 GT sRGB 校准量纲，最后生成搜索伪标签。GT sRGB 不作为最佳曝光真值：

```bash
python main.py calibrate-quality \
  --srgb_root /path/to/SIDD_Small_sRGB_Only \
  --split_manifest outputs/sidd_scene_splits.json \
  --output outputs/quality_calibration.json

python main.py build-pseudo-labels \
  --data_root /path/to/SIDD_Small_Raw_Only \
  --split_manifest outputs/sidd_scene_splits.json \
  --quality_calibration outputs/quality_calibration.json \
  --exposure_offsets=-1.0,0.0,1.0 \
  --output outputs/pseudo_labels_sft.jsonl \
  --audit_output outputs/pseudo_labels_audit.jsonl
```

默认会为每个场景通过快门时间构造等价的 `-1/0/+1` 曝光档状态，避免训练集几乎全部是当前参数标签。`pseudo_labels_audit.jsonl` 保存每个绝对 ISO/快门候选、模拟图像、客观评分和最终选择。

## Training

当前代码不会直接修改 DashScope/千问 API 的模型权重。API 模型只能调用，不能被你的训练脚本反向传播更新。

如果要让 VLM 参数随着任务训练变准，请使用开源 Qwen-VL。项目已经准备了 HuggingFace/Transformers 训练入口，默认使用 LoRA 更新可训练 adapter 参数；服务器显存足够时也可以用 `--no_lora` 做全量微调。

可训练流程是：

1. 用 `build-pseudo-labels` 生成可审计的 `outputs/pseudo_labels_sft.jsonl`。
2. 把项目代码、`outputs/pseudo_labels_sft.jsonl`、相关 preview 图片拷到训练服务器。
3. 在服务器安装训练依赖：

```bash
pip install -e ".[train]"
```

4. 训练开源 Qwen-VL：

```bash
python main.py train-qwen-vl \
  --train_jsonl outputs/pseudo_labels_sft.jsonl \
  --eval_jsonl outputs/pseudo_labels_sft.jsonl \
  --train_split train \
  --eval_split validation \
  --image_root . \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir checkpoints/qwen_vl_exposure_lora \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8
```

训练目标仍然是严格 JSON：

```json
{
  "quality": {
    "brightness": 0.42,
    "noise": 0.18,
    "motion_blur": 0.12,
    "highlight": 0.08,
    "shadow": 0.31,
    "overall_quality": 0.73
  },
  "action": {
    "ISO": 400,
    "Shutter": 0.025
  },
  "continue": true,
  "reason": "local_search_best"
}
```

如果已经有 `outputs/memory.jsonl`，也可以重新导出训练数据：

```bash
python main.py export-training \
  --memory_path outputs/memory.jsonl \
  --output outputs/training_sft.jsonl
```

更完整的远端训练流程见 `docs/TRAINING.md`。

训练完成后，可以在服务器上改成加载本地开源模型和 LoRA：

```bash
EXPOSURE_BACKEND=local_qwen_vl
EXPOSURE_LOCAL_QWEN_VL_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
EXPOSURE_LOCAL_QWEN_VL_ADAPTER_PATH=checkpoints/qwen_vl_exposure_lora
```

## Test

```bash
.venv/bin/python -m pytest
```

模型输出和伪标签的 Action 误差可用以下命令统计：

```bash
python main.py evaluate-actions \
  --predictions outputs/test_predictions.jsonl \
  --targets outputs/pseudo_labels_sft.jsonl \
  --split test \
  --output outputs/action_metrics.json
```

闭环运行结果可用以下命令统计质量增益、成功率、平均轮数和错误提前停止率：

```bash
python main.py evaluate-agent \
  --runs outputs/eval_agent_runs.jsonl \
  --split test \
  --output outputs/agent_metrics.json
```

# Open-Source Qwen-VL Training Workflow

DashScope/千问 API 只能推理，不能被项目代码反向传播更新。要实现“VLM 参数逐步更新，使曝光 Action 数值更准”，需要在服务器上加载开源 Qwen-VL 权重训练。

当前项目已经准备好：

- `loop` 模式：运行固定特征、两次 VLM、train-only RAG、局部搜索和最多三轮反馈闭环；
- `build-pseudo-labels`：生成带候选审计记录的 VLM 弱监督样本；
- `python main.py train-qwen-vl`：用 HuggingFace/Transformers + PEFT LoRA 训练开源 Qwen-VL；
- `checkpoints/qwen_vl_exposure_lora`：保存训练后的 LoRA adapter。
- `EXPOSURE_BACKEND=local_qwen_vl`：训练后把开源 Qwen-VL 接回 Agent 推理。

## 1. Local Smoke Test

本地先确认 pipeline 能跑通即可，不需要训练：

```bash
python main.py \
  --dataset sidd \
  --mode loop \
  --max_samples 5 \
  --max_iterations 1
```

确认这些文件存在：

- `outputs/agent_runs.jsonl`
- `outputs/previews/*.png`
- `artifacts/<scene_id>/round_*.png`

`memory.jsonl` 和训练 JSONL 只在存在有效正增益搜索结果时写入，允许为空。

## 2. Build Scene Splits And Search Pseudo Labels

SIDD 不提供最佳曝光 Action 真值。Pipeline v2 使用固定 scene 划分和无参考质量搜索生成弱监督标签：

```bash
python main.py build-splits \
  --data_root /path/to/SIDD_Small_Raw_Only \
  --output outputs/sidd_scene_splits.json
```

正式生成前用 train 划分中的 GT sRGB 校准噪声与清晰度量纲：

```bash
python main.py calibrate-quality \
  --srgb_root /path/to/SIDD_Small_sRGB_Only \
  --split_manifest outputs/sidd_scene_splits.json \
  --output outputs/quality_calibration.json
```

然后生成伪标签：

```bash

python main.py build-pseudo-labels \
  --data_root /path/to/SIDD_Small_Raw_Only \
  --split_manifest outputs/sidd_scene_splits.json \
  --quality_calibration outputs/quality_calibration.json \
  --exposure_offsets=-1.0,0.0,1.0 \
  --output outputs/pseudo_labels_sft.jsonl \
  --audit_output outputs/pseudo_labels_audit.jsonl
```

GT sRGB 只校准指标尺度，不是最佳曝光参数标签。每个场景默认用快门时间生成等价的 `-1/0/+1` 曝光档状态；只有增益至少 `0.02` 的搜索结果和质量已合格的当前参数目标会进入 SFT，全部候选保存在 audit JSONL。

## 3. Copy To Server

拷贝整个项目最省事，至少需要：

- `src/`
- `main.py`
- `pyproject.toml`
- `outputs/pseudo_labels_sft.jsonl`
- `outputs/pseudo_labels_audit.jsonl`
- `outputs/sidd_scene_splits.json`
- `outputs/previews/`
- `artifacts/`

保持目录结构不变，因为 `pseudo_labels_sft.jsonl` 里的图片路径是相对路径。

## 4. Install Training Dependencies

服务器上：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

如果服务器已经有 PyTorch/CUDA 环境，也可以先按服务器 CUDA 版本安装 torch，再安装：

```bash
pip install -e .
pip install transformers peft accelerate qwen-vl-utils
```

## 5. Train Qwen-VL With LoRA

默认训练开源 Qwen2.5-VL：

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
  --max_length 2048 \
  --max_pixels 401408 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

`--max_pixels 401408` 等于 `512 * 28 * 28`，会把高分辨率 SIDD preview
控制在约 512 个视觉 token。增大该值能保留更多细节，但也会增加序列长度和显存占用。

RTX 4090 24GB 推荐先使用 3B + LoRA。需要尝试 7B 时可改为：

```bash
python main.py train-qwen-vl \
  --train_jsonl outputs/pseudo_labels_sft.jsonl \
  --eval_jsonl outputs/pseudo_labels_sft.jsonl \
  --train_split train \
  --eval_split validation \
  --image_root . \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --output_dir checkpoints/qwen_vl_exposure_lora_7b \
  --num_train_epochs 3
```

这会更新 LoRA adapter 参数，不会改原始 base model 权重。论文里可以表述为“parameter-efficient fine-tuning of the VLM”。如果你必须全量更新全部参数，可以加：

```bash
--no_lora
```

但全量微调需要更多显存。

## 6. What The Model Learns

每条训练样本形式是：

```text
原图 + 可选上一轮结果图 + 固定特征 + metadata
    -> 严格 JSON
```

目标 JSON：

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

其中 `ISO` 和 `Shutter` 都是绝对目标值，EV 只作为派生 metadata，不是模型控制量。训练文件同时包含 `initial` 和 `integration` 两类样本：前者训练第一次建议，后者训练结合 RAG 经验后的第二次建议。

## 7. After Training

训练输出在：

```text
checkpoints/qwen_vl_exposure_lora/
```

下一步可以用训练后的模型替换 DashScope API：

```bash
EXPOSURE_BACKEND=local_qwen_vl
EXPOSURE_LOCAL_QWEN_VL_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
EXPOSURE_LOCAL_QWEN_VL_ADAPTER_PATH=checkpoints/qwen_vl_exposure_lora
EXPOSURE_LOCAL_QWEN_VL_MODEL_FAMILY=qwen2_5_vl
```

然后运行：

```bash
python main.py \
  --dataset sidd \
  --mode loop \
  --max_samples 50 \
  --max_iterations 3 \
  --output outputs/eval_agent_runs.jsonl
```

这样 Agent 调用的是本地开源 Qwen-VL + 训练后的 LoRA adapter。

## 8. Evaluate

训练后对比训练前后：

```bash
python main.py \
  --dataset sidd \
  --mode loop \
  --max_samples 50 \
  --max_iterations 3 \
  --output outputs/eval_agent_runs.jsonl
```

模型 Action 指标：

```bash
python main.py evaluate-actions \
  --predictions outputs/eval_predictions.jsonl \
  --targets outputs/pseudo_labels_sft.jsonl \
  --split validation \
  --output outputs/validation_action_metrics.json
```

训练器自动按 validation loss 保存最佳 checkpoint；论文实验应再对各候选 checkpoint 生成 validation prediction，并按目标 ISO MAE、目标快门 `log2(seconds)` MAE 和 continue F1 选择最终 checkpoint。测试集只在最终模型确定后运行一次。

Agent 主要指标：

- `final_objective_quality.quality.overall_quality` 相对首轮输入的增益；
- `iterations` 是否减少；
- 满意状态是否能在三轮以内达到；
- `final_action` 和两次 VLM Action 的差距是否变小；
- Action 是否更少触发参数 clamp。

以上 Agent 指标可直接聚合：

```bash
python main.py evaluate-agent \
  --runs outputs/eval_agent_runs.jsonl \
  --split test \
  --output outputs/test_agent_metrics.json
```

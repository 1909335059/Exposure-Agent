# ExposureAgent Pipeline v3 远程实验报告

实验日期：2026-08-27  
实验性质：小规模端到端验证，不是正式论文效果实验

## 1. 实验目标

本次在 002 机 RTX 4090 上验证当前约定的完整流程：

```text
输入 sRGB 图像、ISO、快门
        ↓
提取固定图像特征和亮度直方图
        ↓
Qwen2.5-VL 第一次给出质量判断和初步建议
        ↓
从 train-only Memory 检索相似经验
        ↓
Qwen2.5-VL 结合初步建议和经验给出半最终建议
        ↓
在半最终建议附近进行两阶段局部搜索
        ↓
Simulator 生成结果图，Evaluator 判断是否满意
        ↓
满意则结束；不满意则把结果反馈给下一轮第一次 VLM，最多 3 轮
```

## 2. 实验环境

| 项目 | 配置 |
| --- | --- |
| 服务器 | AutoDL 002 机 |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |
| 基础模型 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 微调方式 | LoRA |
| LoRA 可训练参数 | 37,152,768，占总参数 0.9798% |
| 训练轮数 | 3 epoch，24 optimizer steps |
| 代码测试 | 52 passed，1 个第三方弃用 warning |

为减少本地到服务器的传输量，本次从完整 160 scene 清单中按固定种子 42 划分场景，再选取 16 个官方 SIDD GT sRGB 场景，并等比例缩放到宽 1024 像素。该处理适合验证 VLM Pipeline，但不能替代完整分辨率正式实验。

## 3. 数据与训练

16 个场景各构造 `-1/0/+1 EV` 三种输入状态，共 48 条可审计搜索伪标签：

| 项目 | 数值 |
| --- | ---: |
| train | 30 |
| validation | 9 |
| test | 9 |
| 有效标签 | 48 / 48 |
| Action 相对输入发生变化 | 12 |
| 伪标签平均质量增益 | 0.02566 |
| 伪标签最大质量增益 | 0.19117 |

SFT Prompt 已与真实推理 Prompt 对齐，输出目标为包含 `quality`、绝对 `ISO`、绝对 `Shutter` 和 `continue` 的严格 JSON。训练结果如下：

| Epoch | Validation loss |
| ---: | ---: |
| 1 | 0.03053 |
| 2 | 0.02179 |
| 3 | 0.01978 |

最终 train loss 为 `0.06019`，最佳 checkpoint 为 `checkpoint-24`。训练日志中的梯度范数为非零值，例如 `0.2231` 和 `0.1161`，说明 LoRA 参数确实经过反向传播更新。

## 4. RAG 数据检查

普通追加式 Memory 最终保存 7 条 `quality_gain >= 0.02` 的 train 经验，来自 6 个训练场景：

```text
0001_001, 0003_001, 0005_001, 0006_001, 0012_001, 0014_001
```

正式测试场景为 `0007_001`，不在 Memory 中。Memory 中全部记录的 `dataset_split` 都是 `train`，本次验证不存在同 scene 泄漏，测试运行也没有写回 Memory。

## 5. 输入与最终图像

| 输入图像 | 最终结果图像 |
| --- | --- |
| ![输入图像](input_image.png) | ![最终结果图像](final_output_image.png) |

输入为 SIDD test scene `0007_001`：ISO 100、快门 `0.01 s`（1/100 s）、相对 EV `6.6439`。最终图保持 ISO 100，将快门调整为 `0.0125 s`（1/80 s），曝光时间增加 25%，即增加 `0.3219 stop`。

目视上最终图比输入图略亮，构图和颜色保持一致，没有明显新增噪声、色偏或裁切。由于调整幅度较小，两张图的视觉差异也较小。

## 6. 第一轮结构化结果

### 6.1 VLM 初步建议

```json
{
  "quality": {
    "brightness": 0.20868,
    "noise": 0.0,
    "motion_blur": 0.04325,
    "highlight": 0.0,
    "shadow": 0.01517,
    "overall_quality": 0.98831
  },
  "action": {
    "ISO": 100,
    "Shutter": 0.01
  },
  "continue": false,
  "reason": "objective_quality_accepted_current_target"
}
```

Qwen 初步判断当前曝光已经可接受，建议保持原参数。

### 6.2 RAG 检索结果

RAG top-1 来自 train scene `0001_001`，检索距离为 `0.15470`。该经验的初始参数是 ISO 100、1/120 s，成功 Action 是 ISO 100、1/75 s，历史质量增益为 `0.02314`。

检索时使用了固定视觉描述、亮度直方图、质量特征、曝光参数和第一次 VLM Action，并明确排除了当前 test scene。

### 6.3 VLM 综合建议

第二次 Qwen 输出仍为 ISO 100、1/100 s，与第一次建议相同。这说明 RAG 内容成功进入 Prompt，但在本样本中没有改变 VLM 的半最终 Action。

### 6.4 局部搜索与最终 Action

局部搜索围绕半最终建议执行了 16 个候选，选择：

```json
{
  "ISO": 100,
  "Shutter": 0.0125
}
```

搜索分辨率下的质量分数由 `0.98831` 提升到 `0.99127`，搜索增益为 `0.00296`。按完整输出图重新评价后，最终质量为 `0.99076`，相对原图增益为 `0.00244`。

## 7. 停止结果

最终客观指标：

| 指标 | 输入 | 最终 |
| --- | ---: | ---: |
| brightness | 0.20868 | 0.23437 |
| shadow | 0.01517 | 0.00218 |
| highlight | 0.00000 | 0.00000 |
| dynamic range | 0.24984 | 0.27418 |
| motion blur | 0.04325 | 0.04404 |
| overall quality | 0.98831 | 0.99076 |

第一轮输出满足质量门槛，因此停止原因是 `quality_satisfactory`，实际运行 1 轮。最多 3 轮的反馈分支本次没有被触发；这符合流程定义，但本次实验不能证明真实 Qwen 在第二、三轮反馈下的效果。

Agent 汇总指标为：成功率 `1.0`、平均轮数 `1.0`、ISO 绝对变化 `0`、快门变化 `0.3219 stop`、最终不满意率 `0`。

## 8. 实验分析

### 8.1 已验证的能力

- 真实 Qwen2.5-VL-3B LoRA 训练和 checkpoint 重载成功；
- Qwen 两次调用均输出了可被严格 parser 接受的完整 JSON；
- RAG 能检索 train-only 相似经验，并排除测试 scene；
- 两阶段局部搜索实际评价 16 个候选并改变了最终参数；
- Simulator 生成了独立结果图，Evaluator 完成质量门控；
- 输入图、每轮结果、最终图和完整结构化记录均已保存。

### 8.2 当前不能得出的结论

本次只有 1 个 test scene，不能证明模型泛化能力、Action 数值准确率或 RAG 的平均收益。输入本身已被 Evaluator 判为可接受，最终增益只有 `0.00244`，低于 Memory 写入阈值 `0.02`，因此这不是一个强曝光修复案例。

### 8.3 RAG 尚未产生可观察的决策增益

第一次和第二次 Qwen Action 完全一致，最终变化来自局部搜索。因此本次只能证明 RAG 检索和 Prompt 接线正确，不能证明 Qwen 有效利用了经验。后续应在多个欠曝或过曝 test 样本上对比 `VLM` 与 `VLM+RAG` 的 Action 和最终质量。

### 8.4 当前属于 evaluator-assisted VLM

Prompt 中包含亮度直方图、视觉描述和 `objective_quality`，本次 Qwen 返回的 quality 与输入客观指标一致。因此论文中应把当前系统描述为“图像特征与客观质量辅助的 VLM 决策”，不能表述为 VLM 独立完成无参考质量评估。

### 8.5 Simulator 仍是近似模型

最终图是根据 ISO/快门曝光比例模拟得到的，不是相机重新拍摄结果。搜索伪标签也来自相同 Simulator 和 Evaluator，所以可能对该评价函数过拟合。正式论文必须单独说明这一限制，并尽量增加真实多曝光序列或相机实拍验证。

## 9. 本次发现并修复的问题

1. Validation 默认 batch size 为 8，但 VLM collator 只支持 1。现已固定 `per_device_eval_batch_size=1`。
2. `max_length=1536` 会截断全部 assistant 标签，造成 `loss=0` 和 `eval_loss=nan`。现已改为 4096，并在可学习 token 为 0 时立即报错。
3. 原 SFT Prompt 与真实推理 Prompt 不一致。现已统一使用同一组 Prompt builder，并重新完成有效训练。
4. 首次未对齐模型遗漏 `quality` 字段。现已增加最多 3 次严格 schema 修复；仍不合法时继续报错，不填入伪造默认值。
5. 通用 `run` CLI 原先不能显式标记 scene 和 split。现已增加 `--scene_id` 与 `--dataset_split`，保证测试隔离和报告可追踪。
6. 搜索在最长边 768 像素上评分，最终图在 1024 像素上复评，因此搜索增益 `0.00296` 与最终增益 `0.00244` 略有差异。正式评价应统一搜索和最终评分分辨率。

## 10. 结论与下一步

本次 Pipeline v3 小规模实验成功。与上次实验相比，错误的 VLM `continue=false` 已不会跳过 RAG 和局部搜索；scene 泄漏已排除；LoRA 训练不再是 1 step smoke test，而是完成了 3 epoch 的有效反向传播。

下一步应优先运行至少 10 个互不重复的 test scene，并加入人为欠曝和过曝状态，统计 JSON 合法率、Action 误差、质量增益、成功率、平均轮数和错误提前停止率。随后再进行 Base VLM、LoRA VLM、VLM+RAG、VLM+RAG+搜索四组消融，才能形成论文中的效果结论。

## 11. 结果文件

- [完整结构化报告](test_structured_report_aligned.json)
- [原始 AgentResult JSONL](test_agent_run_aligned.jsonl)
- [Agent 指标](test_agent_metrics.json)
- [训练状态](trainer_state.json)
- [场景划分清单](sidd_scene_splits_full.json)
- [伪标签审计](pseudo_labels_audit_aligned.jsonl)
- [LoRA SFT 数据](pseudo_labels_sft_aligned.jsonl)
- [train-only Memory](train_memory_aligned.jsonl)

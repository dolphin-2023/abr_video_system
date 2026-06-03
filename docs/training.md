# 训练与评估

## 前置条件

1. 安装 Python 依赖。
2. 设置 `ABR_LLM_PATH` 指向兼容的本地 Hugging Face 模型目录。
3. 使用 `trace_tools/` 中的脚本生成所需的 trace 划分。

Trace 数据、基座模型、checkpoint 和运行输出均不分发在本仓库中。

## 查看训练计划

默认配方为内部 trace 配方：

```bash
python agent/train.py --stage plan --run-id plan_check
```

外部混合配方也可通过指定配置文件查看：

```bash
python agent/train.py \
  --config agent/configs/train_external_mix_selected_expert_window5_safety.yaml \
  --stage plan \
  --run-id external_mix_plan
```

## 从头运行训练

```bash
python agent/train.py --stage all --run-id local_selected_full
```

已提交的配方按以下流程执行：

```text
collect（收集专家数据） → sft（监督微调） → offline_rl（离线强化学习） → eval（评估）
```

分阶段执行时，请使用相同的 `run-id` 以确保数据一致性。

## 输出产物

每次运行都是自包含的，存放在：

```text
agent/runs/<run_id>/
  config.yaml        本次运行的完整配置
  manifest.json      运行元数据
  data/              训练数据与统计
  checkpoints/       训练 checkpoint
  eval/              评估结果
  logs/              训练日志
```

训练配方也可将 checkpoint 和对应的训练统计同步到 `agent/models/` 下与分布对应的路径。

## 发布在线模型

在线推理服务按以下固定路径搜索模型：

```text
agent/models/netllm_rl/
agent/models/netllm_offline_rl/
agent/models/netllm_sft/
agent/models/training_stats.json
```

发布模型时，必须将 checkpoint 与同一次训练产生的 `training_stats.json` 一起部署。
来自不同 trace 分布的 checkpoint 和 return-to-go 统计数据互不兼容。

---

# Training and Evaluation

## Prerequisites

1. Install the Python dependencies.
2. Set `ABR_LLM_PATH` to a compatible local Hugging Face model directory.
3. Generate the required trace splits with the scripts in `trace_tools/`.

Trace data, base models, checkpoints, and run outputs are not distributed in
this repository.

## Inspect a Training Plan

The default recipe is the internal trace recipe:

```bash
python agent/train.py --stage plan --run-id plan_check
```

The external mix recipe can be inspected with:

```bash
python agent/train.py \
  --config agent/configs/train_external_mix_selected_expert_window5_safety.yaml \
  --stage plan \
  --run-id external_mix_plan
```

## Run from Scratch

```bash
python agent/train.py --stage all --run-id local_selected_full
```

The committed recipes execute:

```text
collect -> sft -> offline_rl -> eval
```

Use the same `run-id` when executing stages separately.

## Outputs

Each run is self-contained under:

```text
agent/runs/<run_id>/
  config.yaml
  manifest.json
  data/
  checkpoints/
  eval/
  logs/
```

The recipe may also synchronize checkpoints and matching training statistics to
distribution-specific paths under `agent/models/`.

## Publishing an Online Model

The online service searches these fixed paths:

```text
agent/models/netllm_rl/
agent/models/netllm_offline_rl/
agent/models/netllm_sft/
agent/models/training_stats.json
```

Publish a checkpoint together with the `training_stats.json` produced by the
same run. A checkpoint and return-to-go statistics from different trace
distributions are not compatible.

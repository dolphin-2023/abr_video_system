# Agent 核心模块

`agent/` 包含 ABR 模拟环境、策略算法、训练流水线、评估工具以及在线决策 API。
这是整个项目的核心，借鉴 NetLLM（Decision Transformer 风格的 LLM 驱动 ABR 策略）实现，使用 Qwen 作为基座模型，通过 LoRA 微调。

## 主要入口

```bash
# 查看训练计划（不实际训练）
python agent/train.py --stage plan --run-id plan_check

# 执行完整训练流水线
python agent/train.py --stage all --run-id local_selected_full
```

可选训练阶段：

```text
plan        仅打印训练计划，不执行
collect     收集专家轨迹数据（BOLA/MPC/BBA）
sft         监督微调（行为克隆）
offline_rl  离线强化学习（return-conditioned 微调）
rl          在线 REINFORCE 微调（带安全门控）
pensieve    训练 PyTorch 版 Pensieve 基线
eval        评估所有已注册策略
all         执行 collect → sft → offline_rl → eval 全流程
```

## 已提交的训练配方

```text
configs/train_local_selected_expert_window5_safety.yaml     （内部 trace）
configs/train_external_mix_selected_expert_window5_safety.yaml （外部混合 trace）
```

两个配方都从专家数据收集开始，使用独立的 checkpoint 和训练统计路径。
本地训练使用内部 `real_world_split` trace，外部混合配方使用生成的 `external_mix_*` trace。
**注意：**来自不同 trace 分布的 checkpoint 和 return-to-go 统计数据不可混用。

## 核心文件

```text
abr_env.py              ABR 离线模拟环境（6×6 状态矩阵，Pensieve 风格）
network.py              NetLLMABR：Decision Transformer ABR 策略网络
data_collector.py       BOLA、MPC、BBA 专家数据收集
experience_dataset.py   SFT 和离线 RL 的滑动窗口数据集
sft_trainer.py          监督行为克隆训练器
offline_rl_trainer.py   Return-conditioned 离线 RL（含低缓冲安全损失）
rl_trainer.py           在线 REINFORCE 微调（含基线安全门控）
pensieve_torch.py       自实现的 PyTorch Pensieve Actor-Critic 基线
evaluate.py             统一的离线多策略评估框架
train.py                配置驱动的训练编排器
chunk_sizes.py          从 DASH MPD 解析真实 chunk 大小
checkpoint_utils.py     瘦身 checkpoint 保存/加载（仅保存 LoRA + ABR 头）
return_utils.py         Return-to-go 参数统一管理
trace_utils.py          Trace 分层采样与筛选
training_eval.py        模型策略的验证评估
run_manager.py          实验运行目录管理
settings.py             全局常量与路径配置
```

## 输出产物

每次训练运行都是自包含的，存放在 `runs/<run_id>/` 下：

```text
runs/<run_id>/
  config.yaml          本次运行的完整配置
  manifest.json        运行元数据和阶段记录
  data/                训练数据与统计
  checkpoints/         训练 checkpoint
  eval/                评估结果与验证历史
  logs/train.log       训练日志
```

生成的模型权重存放在 `models/` 目录下，不会被 Git 跟踪。

在线推理服务按固定路径搜索模型：

```text
models/netllm_rl/
models/netllm_offline_rl/
models/netllm_sft/
models/training_stats.json
```

发布模型时，必须将 checkpoint 与同一轮训练产出的 `training_stats.json` 一起部署。

更多细节请参见 [`docs/training.md`](../docs/training.md)。

---

# Agent

`agent/` contains the ABR simulation environment, policies, training pipeline,
evaluation tools, and online decision API.

## Main Entry Points

```bash
python agent/train.py --stage plan --run-id plan_check
python agent/train.py --stage all --run-id local_selected_full
python agent/evaluate.py --help
```

Available training stages:

```text
plan
collect
sft
offline_rl
rl
pensieve
eval
evaluate
all
```

## Committed Recipes

```text
configs/train_local_selected_expert_window5_safety.yaml
configs/train_external_mix_selected_expert_window5_safety.yaml
```

Both recipes start from expert collection and use independent checkpoint and
training-statistics paths. The local recipe uses internal traces; the external
mix recipe uses generated `external_mix_*` traces.

## Key Files

```text
abr_env.py              Offline ABR simulation environment
network.py              NetLLM-style ABR policy
data_collector.py       BOLA, MPC, and BBA expert collection
experience_dataset.py   SFT and offline RL window dataset
sft_trainer.py          Supervised behavior cloning
offline_rl_trainer.py   Return-conditioned offline RL
rl_trainer.py           Optional online RL fine-tuning
pensieve_torch.py       PyTorch Pensieve baseline
evaluate.py             Unified offline evaluation
train.py                Config-driven training orchestration
```

## Outputs

Runs are self-contained under `runs/<run_id>/`. Generated models are written
under `models/` and are ignored by Git.

The online decision service searches fixed model paths such as
`models/netllm_offline_rl/`. Publish a checkpoint together with the
`training_stats.json` from the same run and trace distribution.

See [`docs/training.md`](../docs/training.md) for details.

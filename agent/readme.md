# agent

`agent/` 是 ABR 模型训练、评测和在线决策的核心目录。

## 推荐入口

训练统一走 `train.py` 和配置文件：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage all --run-id train_v3_full
```

可选阶段：

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

默认配置：

```text
agent/configs/train_v3.yaml
```

## 当前训练链路

```text
collect -> sft -> offline_rl -> rl -> pensieve -> eval
```

`collect` 会从 train split 中按来源分层抽取 validation traces，并在专家数据采集时排除这些 trace。专家数据包含 mixed/BOLA/MPC 和 high-bandwidth/high-MPC 数据，最后合并成 `expert_combined.npz`。

`sft` 使用专家数据做行为克隆，并保存 validation QoE 最好的 checkpoint。

`offline_rl` 从 SFT checkpoint 初始化，使用 NetLLM/Decision Transformer 风格的 return-conditioned 离线训练。默认训练 `state_encoder`、state/action/return/timestep embeddings、LayerNorm、action head 和 LoRA 相关参数，训练后同步到：

```text
agent/models/netllm_offline_rl/
```

`rl` 如果发现 offline RL checkpoint 存在，会优先从 `agent/models/netllm_offline_rl/` 启动；否则回退到 SFT。

`pensieve` 是 PyTorch 复现的 Pensieve baseline，与 NetLLM 共用同一个 `ABREnv`、trace、chunk size 和 QoE，checkpoint 同步到：

```text
agent/models/pensieve_torch/
```

## 主要文件

```text
abr_env.py              ABR 仿真环境
network.py              NetLLM-style ABR 模型
data_collector.py       专家轨迹采集
merge_expert_data.py    专家数据合并
experience_dataset.py   SFT/offline RL 窗口数据集
sft_trainer.py          SFT 行为克隆训练
offline_rl_trainer.py   NetLLM-style offline RL
rl_trainer.py           在线 RL 微调
pensieve_torch.py       PyTorch Pensieve 网络和 checkpoint 工具
pensieve_trainer.py     PyTorch Pensieve 训练
evaluate.py             统一离线评测入口
training_eval.py        训练过程中的统一 validation
trace_utils.py          trace 抽样和分组工具
checkpoint_utils.py     NetLLM checkpoint 保存/加载
run_manager.py          run 目录管理
```

## 输出目录

每次 run 的输出统一进入：

```text
agent/runs/<run_id>/
```

典型结构：

```text
config.yaml
manifest.json
data/
checkpoints/
eval/
logs/
```

正式模型目录：

```text
agent/models/netllm_sft/
agent/models/netllm_offline_rl/
agent/models/netllm_rl/
agent/models/pensieve_torch/
```

## 评测策略

`evaluate.py` 支持：

```text
bola
mpc
random
pensieve
netllm-sft
netllm-offline-rl
netllm-rl
all
```

外部测试 split：

```text
simulator/traces/external_puffer_recent/
simulator/traces/external_weak_mobile/
```

外部 split 由 `origin_traces/import_external_datasets.py` 生成，只用于评测，不参与训练。

## 快速检查

```powershell
D:\Software\miniforge3\envs\myenv\python.exe -m compileall -q agent origin_traces
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan --run-id plan_check
```

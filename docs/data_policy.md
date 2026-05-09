# Data Policy

## Do Not Delete

这些目录是数据来源或实验 trace 池，清理缓存时不要删除：

```text
origin_traces/
simulator/traces/real_world_split/
simulator/traces/real_world/
```

`origin_traces/` 保存原始带宽 trace，`simulator/traces/` 保存 ABR 环境可直接读取的整理后 trace。论文复现实验时需要说明它们的来源、切分方式和用途。

## Safe To Regenerate

这些目录是运行时产物，可以由脚本或训练流程重新生成：

```text
__pycache__/
agent/runs/
agent/logs/
processor/uploads/
processor/dash_output/
```

模型 checkpoint 是否删除取决于是否已经备份。需要连模型一起清理时使用：

```powershell
.\scripts\clean_generated.ps1 -IncludeModels
```

不带 `-IncludeModels` 时，清理脚本不会删除 `agent/models/` 里的正式模型和配置 JSON。

正式模型目录包括：

```text
agent/models/netllm_sft/
agent/models/netllm_offline_rl/
agent/models/netllm_rl/
agent/models/pensieve_torch/
```

# 训练结果与线上模型管理

项目里有两类目录，含义不一样：

```text
agent/runs/<run_id>/      每次实验自己的完整结果
agent/models/             当前在线服务默认加载的正式模型
```

## 每次训练结果在哪里

运行：

```bash
python agent/train.py --stage sft --run-id my_run
```

会生成：

```text
agent/runs/my_run/
```

里面的关键文件是：

```text
data/                         专家数据、training_stats.json、validation_traces.txt
checkpoints/                  本次 run 的 sft/offline_rl/rl/pensieve checkpoint
eval/                         评估 JSON 和 validation history
eval/evaluation_index.json    本次 run 已生成的评估文件索引
logs/train.log                完整训练日志
config.yaml                   本次 run 实际使用的配置快照
manifest.json                 本次 run 的结果索引
```

服务器上找结果优先看：

```bash
ls -lt agent/runs | head
ls agent/runs/<run_id>/eval
tail -n 80 agent/runs/<run_id>/logs/train.log
cat agent/runs/<run_id>/manifest.json
```

## 线上 agent.py 调哪个模型

`agent/agent.py` 不直接读取 `agent/runs/<run_id>`，它读取正式模型目录：

```text
agent/models/netllm_rl/
agent/models/netllm_offline_rl/
agent/models/netllm_sft/
```

启动时优先级是：

```text
netllm_rl -> netllm_offline_rl -> netllm_sft -> random init
```

训练阶段会把最佳 checkpoint 从：

```text
agent/runs/<run_id>/checkpoints/
```

同步到：

```text
agent/models/
```

同时会同步：

```text
agent/runs/<run_id>/data/training_stats.json
```

到：

```text
agent/models/training_stats.json
```

这保证在线服务使用的模型和 return-to-go 统计来自同一轮训练。

当前正式模型来自哪次 run，可以看：

```text
agent/models/active_run.json
```

在线服务启动后，也可以访问：

```text
http://127.0.0.1:8081/health
```

重点看：

```text
active_policy
active_policy_path
target_return
rtg
```

## 评估结果怎么保存

`evaluate.py` 现在会增量写结果。也就是说，即使外部评估跑到一半中断，已经完成的 policy 也会留在目标 JSON 里。

评估文件里有：

```text
complete: true/false
completed_policies: [...]
policies: 汇总指标
episodes: 每条 trace 明细
```

如果 `complete` 是 `false`，说明这个评估文件是中途状态，不是完整实验结论。

## 本地和服务器评估规模

本地默认配置 `agent/configs/train_v3.yaml` 只抽少量外部 trace：

```text
external_puffer_recent: 20
external_weak_mobile: 20
```

服务器配置 `agent/configs/train_server.yaml` 保留完整外部评估：

```text
external_puffer_recent: 300
external_weak_mobile: 252
```

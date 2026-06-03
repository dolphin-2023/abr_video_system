# 系统架构

本项目是一个自适应码率（ABR）视频流媒体的研究原型系统。它整合了离线模拟器、
NetLLM 风格策略、多种基线策略以及本地 DASH 播放演示。

## 系统组件

```text
trace_tools/          原始数据集导入和确定性 trace 划分
agent/                ABR 环境、策略算法、训练和评估
processor/            MP4 上传和 FFmpeg DASH 转码服务
simulator/            Trace 驱动的 HTTP 限速代理
web/                  浏览器播放和指标可视化
```

## 离线训练流程

```text
network traces（网络 trace）
    -> ABREnv（ABR 环境）
    -> BOLA / MPC / BBA 专家数据收集
    -> 窗口级 best-of-expert 选择
    -> 监督微调（SFT）
    -> 离线强化学习（Offline RL）
    -> 评估
```

两个已提交的训练配方是刻意独立设计的：

- `train_local_selected_expert_window5_safety.yaml` 使用内部 `real_world_split` 分布。
- `train_external_mix_selected_expert_window5_safety.yaml` 使用生成的 `external_mix_*` 分布。

每个配方将 checkpoint 和 return-to-go 统计数据写入各自独立的路径。
**请勿**将来自某一分布的 checkpoint 与另一分布的训练统计混用。

## 在线播放流程

```text
浏览器 -> Simulator 代理 -> Processor 提供的 DASH 文件
浏览器 -> Agent 决策 API -> 返回下一块的码率选择
```

在线 Agent 仅加载 `agent/models/` 下已明确发布的模型目录。
训练配方默认不会覆盖这些固定的在线推理路径。

---

# Architecture

The project is a research prototype for adaptive bitrate (ABR) video streaming.
It combines an offline simulator, a NetLLM-style policy, baseline policies, and
a local DASH playback demonstration.

## Components

```text
trace_tools/          Raw dataset import and deterministic trace split creation
agent/                ABR environment, policies, training, and evaluation
processor/            MP4 upload and FFmpeg DASH transcoding service
simulator/            Trace-driven HTTP throttling proxy
web/                  Browser playback and metric visualization
```

## Offline Training Flow

```text
network traces
    -> ABREnv
    -> BOLA / MPC / BBA expert collection
    -> selected expert windows
    -> supervised fine-tuning
    -> offline RL
    -> evaluation
```

The two committed recipes are intentionally independent:

- `train_local_selected_expert_window5_safety.yaml` uses the internal
  `real_world_split` distribution.
- `train_external_mix_selected_expert_window5_safety.yaml` uses the generated
  `external_mix_*` distribution.

Each recipe writes checkpoints and return-to-go statistics to its own paths.
Do not combine a checkpoint from one distribution with training statistics from
another distribution.

## Online Playback Flow

```text
browser -> simulator proxy -> processor DASH files
browser -> agent decision API -> next bitrate
```

The online agent only loads explicitly published model directories under
`agent/models/`. Training recipes do not overwrite those fixed online paths by
default.

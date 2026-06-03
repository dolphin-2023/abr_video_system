# ABR 视频流媒体系统

基于自适应码率（ABR）技术的视频流媒体研究原型系统。集成了 trace 驱动的网络模拟、
NetLLM 风格策略训练、多种基线策略对比、统一评估框架以及本地 DASH 播放演示。

> 原始网络 trace、基座大语言模型、训练 checkpoint、生成的视频文件以及实验运行记录 均不包含在本仓库中。

## 项目特性

- 基于真实网络 trace 的 ABR 模拟环境
- BOLA、MPC、BBA、随机策略以及 PyTorch 仿 Pensieve 基线
- NetLLM 的监督微调（SFT）、离线学习以及在线强化学习
- 确定性的内部和外部 trace 数据准备工具
- 自包含的实验运行目录，便于训练与评估产物的管理
- 本地 MP4 转 DASH 的转码服务、trace 驱动的限速代理以及浏览器播放控制台

## 仓库结构

```text
agent/          ABR 环境、策略算法、训练流水线、评估工具以及在线决策 API
data/           本地原始数据目录，内容不纳入 Git 版本控制
docs/           系统架构、训练流程以及数据来源的文档说明
processor/      MP4 上传和 FFmpeg DASH 转码服务
simulator/      Trace 驱动的 HTTP 限速代理
trace_tools/    原始数据集导入及确定性训练/测试集划分工具
web/            浏览器播放控制台与实时指标可视化
scripts/        维护脚本（如清理缓存与生成产物）
tests/          轻量级单元测试
```

详细项目文档请参阅 [架构说明](docs/architecture.md)、[训练流程](docs/training.md) 和
[数据来源](docs/data_sources.md)。

## 环境安装

需要 Python 3.10 或更高版本。播放演示还需要安装 FFmpeg 和 FFprobe。

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# 或 .venv\Scripts\activate （Windows）

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要 CUDA 4-bit 量化加载基座模型：

```bash
python -m pip install -r requirements-gpu.txt
```

设置本地基座模型目录：

```powershell
# Windows PowerShell
$env:ABR_LLM_PATH = "D:\path\to\your\model"
```

```bash
# Linux / macOS
export ABR_LLM_PATH=/path/to/your/model
```

如果未设置 `ABR_LLM_PATH`，默认使用仓库根目录下的 `models/qwen3.5-4b-base/`。

## 准备网络 Trace

内部数据集的来源为：

[confiwent/Real-world-bandwidth-traces](https://github.com/confiwent/Real-world-bandwidth-traces)

克隆到本地并生成确定性的训练/测试集：

```bash
git clone https://github.com/confiwent/Real-world-bandwidth-traces \
  data/raw/Real-world-bandwidth-traces
python trace_tools/prepare_internal_traces.py --overwrite
```

外部数据集包括 Stanford Puffer、大规模 4G/NB-IoT/5G NSA 测量数据集以及
Beyond Throughput 4G LTE 数据集。本项目选用的 Puffer 数据源为
`2026-05-06 11:00:00 UTC` 至 `2026-05-07 11:00:00 UTC` 的 `video_sent` 日志。

将下载的外部数据文件放入 `data/raw/` 后，生成外部 trace 源池和可选的混合训练集：

```bash
python trace_tools/import_external_datasets.py --clear --max-traces 300 --min-seconds 90
python trace_tools/split_external_for_training.py --overwrite
```

更多细节请参阅 [trace_tools/README.md](trace_tools/README.md) 和
[数据来源说明](docs/data_sources.md)。

## 从头训练

查看默认内部 trace 配方的训练计划（不实际训练）：

```bash
python agent/train.py --stage plan --run-id plan_check
```

执行完整的训练流水线：

```bash
python agent/train.py --stage all --run-id local_selected_full
```

已提交的配方按以下流程执行：

```text
collect（收集专家数据） → sft（监督微调） → offline_rl（离线强化学习） → eval（评估）
```

可用配方：

```text
agent/configs/train_local_selected_expert_window5_safety.yaml
agent/configs/train_external_mix_selected_expert_window5_safety.yaml
```

两个配方将 checkpoint 和训练统计写入各自独立的路径。
请勿将来自某一 trace 分布的 checkpoint 与另一分布的 return-to-go 统计混用。

## 模型与结果

每次训练运行的结果存放在：

```text
agent/runs/<run_id>/
```

重要文件包括：

```text
logs/train.log              完整的训练日志
manifest.json                运行元数据
eval/evaluation_index.json   评估索引
eval/*.json                  各策略评估结果
```

训练 checkpoint 可同步至 `agent/models/` 下与 trace 分布对应的路径。在线推理服务
仅加载明确发布的固定模型路径，详见 [docs/training.md](docs/training.md)。

## 本地播放演示

一键启动 Processor、Agent 决策 API、网络模拟器和 Web 静态服务：

```bash
python start_system.py
```

打开浏览器访问 <http://127.0.0.1:3000/>。通过 Processor API 上传 MP4 视频，
然后使用返回的 `video_id` 在 Web 界面中播放。

## 代码验证

```bash
# 编译检查所有 Python 模块
python -m compileall -q agent processor simulator trace_tools start_system.py

# 运行单元测试
python -m unittest discover -s tests -p "test_*.py"
```

## 清理

清除缓存和生成的运行产物：

```powershell
.\scripts\clean_generated.ps1
```

如需同时清除生成的模型权重，加上 `-IncludeModels` 选项：

```powershell
.\scripts\clean_generated.ps1 -IncludeModels
```

---

# ABR Video System

An adaptive bitrate (ABR) video streaming research prototype with trace-driven
simulation, NetLLM-style training, baseline policies, unified evaluation, and a
local DASH playback demo.

> Raw network traces, base language models, checkpoints, generated videos, and
> experiment runs are intentionally not included in this repository.

## Features

- Pensieve-style ABR simulation environment with real network traces
- BOLA, MPC, BBA, random, and PyTorch Pensieve baselines
- NetLLM-style supervised fine-tuning, offline RL, and optional online RL
- Deterministic internal and external trace preparation tools
- Self-contained run directories for training and evaluation artifacts
- Local MP4-to-DASH transcoding, trace-driven throttling, and browser playback

## Repository Layout

```text
agent/          ABR environment, policies, training, evaluation, and decision API
data/           Local raw dataset location; contents are ignored by Git
docs/           Architecture, training, and data source documentation
processor/      MP4 upload and FFmpeg DASH transcoding service
simulator/      Trace-driven HTTP throttling proxy
trace_tools/    Raw dataset import and deterministic split creation
web/            Browser playback and metric visualization
scripts/        Maintenance scripts
tests/          Lightweight unit tests
```

See [architecture](docs/architecture.md), [training](docs/training.md), and
[data sources](docs/data_sources.md) for the full project notes.

## Installation

Python 3.10 or newer is required. FFmpeg and FFprobe are also required for the
playback demo.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA 4-bit model loading:

```bash
python -m pip install -r requirements-gpu.txt
```

Set the compatible local Hugging Face base model directory:

```powershell
$env:ABR_LLM_PATH = "D:\path\to\your\model"
```

```bash
export ABR_LLM_PATH=/path/to/your/model
```

If `ABR_LLM_PATH` is not set, the default is
`models/qwen3.5-4b-base/` under the repository root.

## Prepare Network Traces

The internal dataset source is
[confiwent/Real-world-bandwidth-traces](https://github.com/confiwent/Real-world-bandwidth-traces).
Clone it locally and generate the internal train/test splits:

```bash
git clone https://github.com/confiwent/Real-world-bandwidth-traces \
  data/raw/Real-world-bandwidth-traces
python trace_tools/prepare_internal_traces.py --overwrite
```

External datasets include Stanford Puffer, the large-scale 4G/NB-IoT/5G NSA
measurement dataset, and the Beyond Throughput 4G LTE dataset. The selected
Puffer source is the `video_sent` log from `2026-05-06 11:00:00 UTC` to
`2026-05-07 11:00:00 UTC`.

After downloading the external files into `data/raw/`, generate the source
pools and optional external mix splits:

```bash
python trace_tools/import_external_datasets.py --clear --max-traces 300 --min-seconds 90
python trace_tools/split_external_for_training.py --overwrite
```

See [trace_tools/README.md](trace_tools/README.md) for the expected local file
layout and [docs/data_sources.md](docs/data_sources.md) for citations.

## Train from Scratch

Inspect the default internal trace recipe without starting training:

```bash
python agent/train.py --stage plan --run-id plan_check
```

Run the complete committed pipeline:

```bash
python agent/train.py --stage all --run-id local_selected_full
```

The committed recipes execute:

```text
collect -> sft -> offline_rl -> eval
```

Available recipes:

```text
agent/configs/train_local_selected_expert_window5_safety.yaml
agent/configs/train_external_mix_selected_expert_window5_safety.yaml
```

The two recipes write to independent checkpoint and training-statistics paths.
Do not mix a checkpoint from one trace distribution with return-to-go
statistics from another distribution.

## Results and Models

Each run is stored under:

```text
agent/runs/<run_id>/
```

Important files include:

```text
logs/train.log
manifest.json
eval/evaluation_index.json
eval/*.json
```

Training checkpoints may be synchronized to distribution-specific paths under
`agent/models/`. The online service only loads explicitly published fixed model
paths; see [docs/training.md](docs/training.md).

## Local Playback Demo

Start the processor, decision API, network simulator, and static web server:

```bash
python start_system.py
```

Open <http://127.0.0.1:3000/>. Upload an MP4 through the processor API, then use
the returned `video_id` in the web interface.

## Validation

```bash
python -m compileall -q agent processor simulator trace_tools start_system.py
python -m unittest discover -s tests -p "test_*.py"
```

## Cleanup

Remove caches and generated runs:

```powershell
.\scripts\clean_generated.ps1
```

Also remove generated model outputs:

```powershell
.\scripts\clean_generated.ps1 -IncludeModels
```

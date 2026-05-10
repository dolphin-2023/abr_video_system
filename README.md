# ABR Video System

## 快速定位结果

每次训练的完整实验结果保存在：

```text
agent/runs/<run_id>/
```

最常看的文件：

```text
agent/runs/<run_id>/logs/train.log
agent/runs/<run_id>/manifest.json
agent/runs/<run_id>/eval/evaluation_index.json
agent/runs/<run_id>/eval/*.json
```

在线 `agent.py` 默认加载的正式模型保存在：

```text
agent/models/netllm_rl/
agent/models/netllm_offline_rl/
agent/models/netllm_sft/
agent/models/training_stats.json
agent/models/active_run.json
```

详细说明见：

```text
docs/result_management.md
docs/server_training.md
```

本地默认外部评估只抽少量 trace；服务器完整训练请使用：

```bash
python agent/train.py --config agent/configs/train_server.yaml --stage plan --run-id server_plan
```

基于智能体的自适应码率视频传输实验系统。当前训练链路是：

```text
collect -> sft -> offline_rl -> rl -> pensieve -> eval
```

项目包含真实网络 trace 仿真、NetLLM-style ABR 模型、SFT、NetLLM-style offline RL、在线 RL、PyTorch Pensieve baseline、统一评测，以及本地上传转码和播放演示系统。

## 目录结构

```text
agent/          ABR 环境、模型、训练、评测和在线决策服务
origin_traces/  原始 trace 数据和外部数据导入工具
simulator/      trace 驱动的限速代理与标准 trace split
processor/      MP4 上传与 DASH 转码服务
web/            播放页面与指标可视化
docs/           数据边界、测试命令和论文写作辅助说明
scripts/        本地维护脚本
tests/          轻量单元测试
```

## 环境

推荐使用你的本地环境：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe --version
D:\Software\miniforge3\envs\myenv\python.exe -m pip install -r requirements.txt
```

默认本地大模型路径：

```text
D:\ai-models\qwen3.5-4b-base
```

需要替换时可以设置：

```powershell
$env:ABR_LLM_PATH="D:\path\to\your\model"
```

## 训练

查看计划，不开始训练：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan --run-id plan_check
```

完整训练：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage all --run-id train_v3_full
```

分阶段训练：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage collect --run-id train_v3_full
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage sft --run-id train_v3_full
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage offline_rl --run-id train_v3_full
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage rl --run-id train_v3_full
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage pensieve --run-id train_v3_full
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage eval --run-id train_v3_full
```

默认配置：

```text
agent/configs/train_v3.yaml
```

训练输出：

```text
agent/runs/<run_id>/
```

正式 checkpoint 会同步或手动放入：

```text
agent/models/netllm_sft/
agent/models/netllm_offline_rl/
agent/models/netllm_rl/
agent/models/pensieve_torch/
```

## 外部测试集

你下载的新外部数据位于：

```text
origin_traces/new/Puffer/
origin_traces/new/A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements/
origin_traces/new/Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics/
```

重新生成外部测试 split：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe origin_traces\import_external_datasets.py --clear --max-traces 300 --min-seconds 90
```

输出：

```text
simulator/traces/external_puffer_recent/   300 条近期 Puffer 分桶抽样 trace
simulator/traces/external_weak_mobile/     252 条弱网/移动网络 trace
```

处理策略：

```text
Puffer: 按 (session_id, index, channel) 分流；delivery_rate 从 bytes/s 转 Kbps；按中位吞吐分桶抽样。
PERFORM: 读取 Current Netw. DL / Mean Netw. DL / 5G PDSCH Throughput / LTE PDSCH Throughput，按 Kbps 处理。
LTE: 只保留 State == D 且 DL_bitrate > 0 的连续下载段。
```

这两个 split 只用于外部测试，不参与训练。

单独评测外部集：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\evaluate.py --trace-split external_puffer_recent --episodes 300 --sample-mode stratified --policies all --output agent\runs\external_eval\puffer_recent_all.json
D:\Software\miniforge3\envs\myenv\python.exe agent\evaluate.py --trace-split external_weak_mobile --episodes 252 --sample-mode stratified --policies all --output agent\runs\external_eval\weak_mobile_all.json
```

## 评测策略

统一评测入口支持：

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

`train.py --stage eval` 会读取配置并评测内部 test、近期 Puffer 外部集、弱网移动外部集。

## 快速检查

```powershell
D:\Software\miniforge3\envs\myenv\python.exe -m compileall -q agent origin_traces processor simulator start_system.py
D:\Software\miniforge3\envs\myenv\python.exe -m unittest discover -s tests -p "test_*.py"
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan --run-id plan_check
```

## 演示系统

```powershell
D:\Software\miniforge3\envs\myenv\python.exe start_system.py
```

浏览器打开：

```text
http://127.0.0.1:3000/
```

## 清理

清理缓存和运行产物：

```powershell
.\scripts\clean_generated.ps1
```

只有确认模型 checkpoint 已备份后，才使用：

```powershell
.\scripts\clean_generated.ps1 -IncludeModels
```

清理脚本不应删除 `origin_traces/` 或 `simulator/traces/`。

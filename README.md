# ABR Video System

基于智能体的自适应码率视频传输演示与实验系统。项目把视频上传转码、网络限速模拟、浏览器播放、NetLLM-style 码率决策、离线训练和评估放在同一个工程里，便于毕设演示和论文实验复现。

## 目录结构

```text
agent/          ABR 智能体、训练、评估和在线决策服务
processor/      MP4 上传与 DASH 转码服务，默认端口 8080
simulator/      带宽 trace 限速代理，默认端口 8082
web/            浏览器播放和指标可视化页面，默认端口 3000
start_system.py 一键启动本地演示服务
```

`simulator/traces/real_world_split/` 保留了已整理的 train/test trace，可直接用于演示和离线评估。原始 trace、上传视频、DASH 输出、训练 run、模型 checkpoint 和论文材料属于本地/实验产物，不纳入 GitHub 仓库。

## 环境准备

建议在单独的 conda 环境中运行：

```powershell
conda activate myenv
pip install -r requirements.txt
```

系统还需要能直接调用：

```text
ffmpeg
ffprobe
```

智能体默认读取本地大模型路径 `D:\ai-models\qwen3.5-4b-base`。如需换路径，可以设置环境变量：

```powershell
$env:ABR_LLM_PATH="D:\path\to\your\model"
```

## 启动演示

```powershell
python start_system.py
```

启动后打开：

```text
http://127.0.0.1:3000/
```

服务端口：

```text
processor  8080  上传 MP4 并转成 DASH
agent      8081  根据播放状态输出下一个码率档位
simulator  8082  按真实 trace 限速代理视频分片
web        3000  播放页面与指标图表
```

## 实验入口

训练和评估入口见 `agent/readme.md`。常用流程：

```powershell
python agent\train.py --stage plan
python agent\train.py --stage all --run-id train_v3_next
```

实验输出统一进入：

```text
agent/runs/<run_id>/
```

正式模型 checkpoint 可放在：

```text
agent/models/netllm_sft/
agent/models/netllm_rl/
```

这些目录已在 `.gitignore` 中忽略，避免把本地大文件误提交到 GitHub。

## 可清理内容

可以随时重新生成并清理：

```text
__pycache__/
processor/uploads/
processor/dash_output/
agent/runs/
agent/logs/
agent/models/netllm_sft/
agent/models/netllm_rl/
origin_traces/
```

当前仓库保留 `agent/models/training_stats.json` 和 `agent/models/chunk_sizes.json`，用于演示和评估时保持 return-to-go 与 chunk size 配置稳定。

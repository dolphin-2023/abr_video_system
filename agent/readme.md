# agent 目录说明

`agent/` 是 ABR 模型训练和推理的核心目录。这里负责网络仿真、专家数据采集、SFT、RL、离线评估，以及演示系统中的在线码率决策。

## 推荐用法

始终使用 `myenv`：

```powershell
cd D:\Desktop\毕设\abr_video_system
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --help
```

现在训练入口已经精简为配置驱动。命令行只负责选择配置、阶段和 run 目录；训练细节都写在配置文件里。

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage all --run-id train_v3_next
```

默认配置文件：

```text
agent/configs/train_v3.yaml
```

如果要分阶段跑同一个实验：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage collect --run-id train_v3_next
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage sft --run-id train_v3_next
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage rl --run-id train_v3_next
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage eval --run-id train_v3_next
```

## 输出目录

新训练输出统一进入 `agent/runs/<run_id>/`，不要再把临时数据直接堆在 `agent/` 根目录。

```text
agent/runs/<run_id>/
├── config.yaml
├── manifest.json
├── data/
├── checkpoints/
├── eval/
└── logs/
```

各目录含义：

```text
data/         专家数据、合并数据、training_stats.json
checkpoints/  本次 run 的 sft/rl checkpoint
eval/         validation history 和 test 评估结果
logs/         train.log
config.yaml   本次 run 实际使用的配置快照
manifest.json 本次 run 的关键路径和 validation trace 列表
```

RL 阶段会额外保留候选 checkpoint，方便分析训练是否学偏：

```text
checkpoints/rl              正式 best safe 模型，评估默认读取它
checkpoints/rl_baseline     RL 开始前的 SFT 保底模型
checkpoints/rl_best_safe    当前 best safe 的显式备份
checkpoints/rl_latest       最近一次验证时的 RL 权重
checkpoints/rl_candidate_*  每次验证保存的候选权重，即使 unsafe 也保留
```

`agent/models/` 保留为正式模型目录，适合放演示系统或论文最终结果要使用的 checkpoint。旧的 `expert_*.npz`、`expert_*.stats.json` 是历史输出，新实验不建议继续写到根目录。

## 配置文件

`configs/train_v3.yaml` 是默认实验配置，主要分成这些段：

```text
run         run 名称、日志文件名、输出根目录
env         LLM 路径、chunk size 路径、ABREnv 日志开关
runtime     CPU 线程等运行时设置
common      QoE profile、trace split、窗口长度
validation  SFT/RL 共用的验证 trace 抽样
data        专家数据采集和合并
sft         SFT 训练参数
rl          RL 微调参数
eval        test 评估参数
```

常改的地方：

```text
env.llm_path
runtime.torch_num_threads
data.general.episodes
data.high.episodes
sft.epochs
rl.episodes
eval.final.episodes
```

## 主要程序

### train.py

唯一推荐的训练调度入口。它只保留四个命令行参数：

```text
--config   训练配置文件，默认 agent/configs/train_v3.yaml
--stage    plan/all/collect/sft/rl/eval/evaluate
--run-id   agent/runs/ 下的 run 名称
--run-dir  显式指定 run 目录
```

`train.py` 做调度，不直接写死训练策略。它会按配置调用数据采集、SFT、RL 和评估，并把所有输出归档到同一个 run 目录。

### evaluate.py

离线评估程序。推荐平时通过 `train.py --stage eval` 调用；需要单独手动比较策略时再直接运行它。

它可以比较：

```text
bola
mpc
random
netllm-sft
netllm-rl
```

### agent.py

演示系统的在线码率决策服务。前端播放时把当前 ABR 状态发给它，它返回下一个码率档位。这个文件服务的是 demo/系统展示，不负责离线训练。

### chunk_sizes.py

真实 DASH chunk size 工具。可以从 DASH 输出目录生成 `chunk_sizes.json`，也可以被 `ABREnv` 读取现成的 chunk size 表。

## 核心训练文件

### abr_env.py

ABR 仿真环境。负责读取 trace、模拟下载时间、维护 buffer、计算 stall/sleep、计算 QoE reward，并支持真实 DASH chunk size。

默认不打印环境细节。需要调试时设置：

```powershell
$env:ABR_ENV_VERBOSE="1"
```

### network.py

NetLLM-style ABR 模型结构。包含状态编码、return/action/timestep embedding、base LLM、LoRA 和 action head。

### data_collector.py

专家轨迹采集。支持 `bola`、`mpc`、`mixed`、`high-mpc`。输出 `.npz` 轨迹文件和 `.stats.json` 统计文件。

通常不需要手动运行它，`train.py --stage collect` 会按配置调用。

### merge_expert_data.py

合并多份专家数据，比如 general 数据和 high-bandwidth 数据。合并后会重新统计动作分布、reward 范围和 return 范围。

### experience_dataset.py

把专家轨迹切成 SFT 训练窗口，并计算训练用 return-to-go。它生成的 `training_stats.json` 会被 SFT validation、RL 和评估复用。

### sft_trainer.py

SFT 行为克隆训练。它从专家轨迹监督学习动作，支持 class weight、QoE validation gate、early stop、最佳 checkpoint 保存和粗粒度进度提示。

### rl_trainer.py

RL 微调。目前是带安全验证门控的 REINFORCE，不是 PPO。它从 SFT checkpoint 出发，默认放开 projection 层和 action head，用小 KL anchor 约束不要离 SFT 太远。

正式模型仍由 validation gate 控制；候选模型会保存在 `rl_candidate_*` 和 `rl_latest`，便于分析 RL 到底是提高 QoE、过度降码率，还是在某些 trace 上卡顿变差。

### training_eval.py

训练过程中的统一验证函数。SFT 和 RL 都用它评估 validation traces，保证指标口径一致。

## 工具文件

### config_utils.py

配置读取工具。负责读取 YAML/JSON，并提供配置段读取、路径解析等小函数。

### run_manager.py

run 目录管理工具。负责创建 `agent/runs/<run_id>/data`、`checkpoints`、`eval`、`logs` 等目录。

### trace_utils.py

trace 抽样和筛选工具。负责 first/random/stratified 抽样，以及 high-bandwidth trace 筛选。

### return_utils.py

统一管理 NetLLM/Decision Transformer 式 return-to-go。当前逻辑是：

```text
剩余目标回报 = 当前目标回报 - 本步处理后的 reward
```

处理后的 reward 尺度来自 `training_stats.json`。

### checkpoint_utils.py

checkpoint 保存和加载工具。新版默认保存瘦身 checkpoint：

```text
adapter_model.*
modules_except_plm.bin
checkpoint_meta.json
```

同时还能读取旧版 `.pth` 大文件。

### settings.py

全局常量和默认路径，例如码率档位、segment 时长、base model 默认路径、正式模型路径。

## 数据和历史文件

这些文件是历史实验输出或正式模型，不是新的推荐写入位置：

```text
agent/expert_*.npz
agent/expert_*.stats.json
agent/logs/
agent/models/evaluation_*.json
agent/models/netllm_sft.pth
agent/models/netllm_rl.pth
```

新的临时训练数据应该进入：

```text
agent/runs/<run_id>/data/
agent/runs/<run_id>/eval/
agent/runs/<run_id>/logs/
```

正式模型可以保留在：

```text
agent/models/netllm_sft/
agent/models/netllm_rl/
```

## 快速检查

```powershell
D:\Software\miniforge3\envs\myenv\python.exe -m py_compile agent\*.py
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --help
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan --run-id plan_check
```

`plan` 阶段只检查配置和创建目录，不会开始训练。检查完可以删除对应的临时 run 目录。

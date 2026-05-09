# Testing

推荐先跑不依赖训练的快速检查：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe -m compileall -q agent processor simulator start_system.py
D:\Software\miniforge3\envs\myenv\python.exe -m unittest discover -s tests -p "test_*.py"
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage plan --run-id plan_check
```

从头训练与评估：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage all --run-id train_v3_full
```

完整 `all` 顺序是：

```text
collect -> sft -> offline_rl -> rl -> pensieve -> eval
```

如果只想确认评估链路，先使用已有 checkpoint 跑：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe agent\train.py --stage eval --run-id eval_check
```

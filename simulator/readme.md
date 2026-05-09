# simulator 目录说明

`simulator/` 是网络带宽模拟代理。它从 processor 取视频分片，再按照真实网络 trace 限速后转发给浏览器。

## 端口

```text
http://127.0.0.1:8082
```

## 文件分工

```text
proxy.py                    # FastAPI 代理服务
traces/real_world_split/    # 已整理好的 train/test trace
```

## 启动

```powershell
conda activate myenv
cd D:\Desktop\毕设\abr_video_system\simulator
python -m uvicorn proxy:app --reload --host 127.0.0.1 --port 8082
```

## 常用接口

查看当前 trace 和带宽：

```text
GET http://127.0.0.1:8082/api/sim/status
```

随机切换 trace：

```text
POST http://127.0.0.1:8082/api/sim/random_trace
```

视频代理路径：

```text
http://127.0.0.1:8082/video/{video_id}/{video_id}.mpd
```

## trace split

默认前端演示使用 `test` split。可以用环境变量切换：

```powershell
$env:ABR_SIM_TRACE_SPLIT="train"
```

也可以直接指定自定义目录：

```powershell
$env:ABR_SIM_TRACE_DIR="D:\path\to\traces"
```

## 和 agent 训练的关系

`simulator/` 用于在线播放演示；`agent/abr_env.py` 用于离线训练和评估。两者都使用真实 trace，但用途不同：

```text
simulator/proxy.py  浏览器真实播放时限速
agent/abr_env.py    训练/评估时快速仿真
```

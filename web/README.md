# Web 前端

`web/` 包含浏览器播放页面。它展示视频画面、缓冲区水位、吞吐量和 QoE 等实时指标，
同时向 Agent 请求码率决策。

## 启动服务

```bash
python -m http.server 3000 --directory web
```

打开 <http://127.0.0.1:3000/>。

播放页依赖以下后端服务：

```text
processor  http://127.0.0.1:8080    （视频上传与 DASH 转码）
agent      http://127.0.0.1:8081    （ABR 决策 API）
simulator  http://127.0.0.1:8082    （trace 驱动的网络限速代理）
```

在仓库根目录执行 `python start_system.py` 可一键启动全部本地服务。

---

# Web

`web/` contains the browser playback page. It displays video, buffer,
throughput, and QoE metrics while requesting bitrate decisions from the agent.

## Service

```bash
python -m http.server 3000 --directory web
```

Open <http://127.0.0.1:3000/>.

The page expects:

```text
processor  http://127.0.0.1:8080
agent      http://127.0.0.1:8081
simulator  http://127.0.0.1:8082
```

Run `python start_system.py` from the repository root to start all local
services.

# web 目录说明

`web/` 是浏览器端播放页面，负责展示视频、buffer、吞吐量、QoE 曲线，并把当前播放状态发给智能体服务。

## 启动

```powershell
conda activate myenv
cd D:\Desktop\毕设\abr_video_system\web
python -m http.server 3000
```

打开：

```text
http://127.0.0.1:3000/
```

## 依赖的后端服务

页面默认访问：

```text
agent      http://127.0.0.1:8081/abr_decision
simulator  http://127.0.0.1:8082
```

所以使用页面前，至少要启动：

```text
processor  8080
agent      8081
simulator  8082
web        3000
```

最简单方式是在项目根目录运行：

```powershell
python start_system.py
```

## 播放流程

1. 先用 processor 上传 mp4，拿到 `video_id`。
2. 在网页输入 `video_id`。
3. 页面通过 simulator 代理加载 DASH 视频。
4. 每个视频分片下载完成后，页面把 buffer、吞吐量、当前码率、上一段 reward 发给 agent。
5. agent 返回下一段建议码率，页面调用 dash.js 切换档位。

## 注意事项

`index.html` 里的在线 QoE 只是演示用指标，论文结果以 `agent/evaluate.py` 的离线评估为准。

如果浏览器控制台出现跨域错误，确认三个 FastAPI 服务都已经启动，并且端口没有被占用。

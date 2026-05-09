# processor 目录说明

`processor/` 是视频上传与 DASH 转码服务。它接收 mp4，调用 `ffmpeg` 转成多码率 DASH 文件，供前端和网络模拟器播放。

## 端口

```text
http://127.0.0.1:8080
```

## 文件分工

```text
app.py        # FastAPI 服务，提供 /upload 和 /video 静态文件
uploads/      # 上传的原始 mp4
dash_output/  # 转码后的 mpd/m4s
```

## 启动

```powershell
conda activate myenv
cd D:\Desktop\毕设\abr_video_system\processor
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8080
```

## 上传视频测试

假设 `uploads\test.mp4` 存在：

```powershell
curl.exe -X POST "http://127.0.0.1:8080/upload" -H "accept: application/json" -F "file=@uploads\test.mp4"
```

返回值里会有：

```text
video_id
play_url
```

如果 `video_id` 是 `abc123`，DASH 地址就是：

```text
http://127.0.0.1:8080/video/abc123/abc123.mpd
```

实际演示时不要直接让前端访问 8080，而是通过 simulator 的 8082 代理访问，这样才能模拟带宽波动。

## 依赖

需要系统里能直接调用：

```text
ffmpeg
ffprobe
```

如果命令不存在，请先把 FFmpeg 加到 Windows PATH。

## 可以清理吗

`uploads/` 和 `dash_output/` 都是可再生成文件。空间不够时可以清理，但清理后对应视频就不能播放，需要重新上传转码。

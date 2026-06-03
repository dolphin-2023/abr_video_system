# Processor 视频处理服务

`processor/` 是 MP4 上传与 DASH 转码服务。它调用 FFmpeg 将上传的视频转码为多个码率档位，供浏览器播放演示使用。

## 启动服务

```bash
python -m uvicorn app:app --app-dir processor --host 127.0.0.1 --port 8080
```

系统依赖：

```text
ffmpeg
ffprobe
```

生成的上传文件和 DASH 输出分别存放在 `uploads/` 和 `dash_output/` 目录中。
这两个目录不会被提交到 Git，可以随时重新生成。

---

# Processor

`processor/` is the MP4 upload and DASH transcoding service. It calls FFmpeg to
create multiple bitrate representations for the browser demo.

## Service

```bash
python -m uvicorn app:app --app-dir processor --host 127.0.0.1 --port 8080
```

Required system commands:

```text
ffmpeg
ffprobe
```

Generated uploads and DASH files are stored in `uploads/` and `dash_output/`.
Both directories are ignored by Git and can be regenerated.

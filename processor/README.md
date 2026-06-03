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

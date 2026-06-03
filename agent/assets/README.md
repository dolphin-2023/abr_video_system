# Agent 资源文件

`example_chunk_sizes.json` 是一个示例 DASH chunk 大小表，供提交的训练配方使用。
它允许模拟器在不依赖实际视频文件的情况下，模拟可变的片段大小。

为你的转码视频生成 chunk 大小表：

```bash
python agent/chunk_sizes.py \
  --video-id <video_id> \
  --output agent/assets/chunk_sizes.local.json
```

然后在本地训练配置中设置 `env.chunk_size_path`。以 `.local.json` 结尾的文件会被 Git 忽略。

---

# Agent Assets

`example_chunk_sizes.json` is a small example DASH chunk-size table used by the
committed training recipes. It allows the simulator to model variable segment
sizes without committing video files.

Generate a table for your own transcoded video with:

```bash
python agent/chunk_sizes.py \
  --video-id <video_id> \
  --output agent/assets/chunk_sizes.local.json
```

Then set `env.chunk_size_path` in a local training config. Files ending in
`.local.json` are ignored by Git.

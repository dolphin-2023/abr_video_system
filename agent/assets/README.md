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

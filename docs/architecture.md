# Architecture

The project is a research prototype for adaptive bitrate (ABR) video streaming.
It combines an offline simulator, a NetLLM-style policy, baseline policies, and
a local DASH playback demonstration.

## Components

```text
trace_tools/          Raw dataset import and deterministic trace split creation
agent/                ABR environment, policies, training, and evaluation
processor/            MP4 upload and FFmpeg DASH transcoding service
simulator/            Trace-driven HTTP throttling proxy
web/                  Browser playback and metric visualization
```

## Offline Training Flow

```text
network traces
    -> ABREnv
    -> BOLA / MPC / BBA expert collection
    -> selected expert windows
    -> supervised fine-tuning
    -> offline RL
    -> evaluation
```

The two committed recipes are intentionally independent:

- `train_local_selected_expert_window5_safety.yaml` uses the internal
  `real_world_split` distribution.
- `train_external_mix_selected_expert_window5_safety.yaml` uses the generated
  `external_mix_*` distribution.

Each recipe writes checkpoints and return-to-go statistics to its own paths.
Do not combine a checkpoint from one distribution with training statistics from
another distribution.

## Online Playback Flow

```text
browser -> simulator proxy -> processor DASH files
browser -> agent decision API -> next bitrate
```

The online agent only loads explicitly published model directories under
`agent/models/`. Training recipes do not overwrite those fixed online paths by
default.

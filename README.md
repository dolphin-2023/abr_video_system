# ABR Video System

An adaptive bitrate (ABR) video streaming research prototype with trace-driven
simulation, NetLLM-style training, baseline policies, unified evaluation, and a
local DASH playback demo.

> Raw network traces, base language models, checkpoints, generated videos, and
> experiment runs are intentionally not included in this repository.

## Features

- Pensieve-style ABR simulation environment with real network traces
- BOLA, MPC, BBA, random, and PyTorch Pensieve baselines
- NetLLM-style supervised fine-tuning, offline RL, and optional online RL
- Deterministic internal and external trace preparation tools
- Self-contained run directories for training and evaluation artifacts
- Local MP4-to-DASH transcoding, trace-driven throttling, and browser playback

## Repository Layout

```text
agent/          ABR environment, policies, training, evaluation, and decision API
data/           Local raw dataset location; contents are ignored by Git
docs/           Architecture, training, and data source documentation
processor/      MP4 upload and FFmpeg DASH transcoding service
simulator/      Trace-driven HTTP throttling proxy
trace_tools/    Raw dataset import and deterministic split creation
web/            Browser playback and metric visualization
scripts/        Maintenance scripts
tests/          Lightweight unit tests
```

See [architecture](docs/architecture.md), [training](docs/training.md), and
[data sources](docs/data_sources.md) for the full project notes.

## Installation

Python 3.10 or newer is required. FFmpeg and FFprobe are also required for the
playback demo.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA 4-bit model loading:

```bash
python -m pip install -r requirements-gpu.txt
```

Set the compatible local Hugging Face base model directory:

```powershell
$env:ABR_LLM_PATH = "D:\path\to\your\model"
```

```bash
export ABR_LLM_PATH=/path/to/your/model
```

If `ABR_LLM_PATH` is not set, the default is
`models/qwen3.5-4b-base/` under the repository root.

## Prepare Network Traces

The internal dataset source is
[confiwent/Real-world-bandwidth-traces](https://github.com/confiwent/Real-world-bandwidth-traces).
Clone it locally and generate the internal train/test splits:

```bash
git clone https://github.com/confiwent/Real-world-bandwidth-traces \
  data/raw/Real-world-bandwidth-traces
python trace_tools/prepare_internal_traces.py --overwrite
```

External datasets include Stanford Puffer, the large-scale 4G/NB-IoT/5G NSA
measurement dataset, and the Beyond Throughput 4G LTE dataset. The selected
Puffer source is the `video_sent` log from `2026-05-06 11:00:00 UTC` to
`2026-05-07 11:00:00 UTC`.

After downloading the external files into `data/raw/`, generate the source
pools and optional external mix splits:

```bash
python trace_tools/import_external_datasets.py --clear --max-traces 300 --min-seconds 90
python trace_tools/split_external_for_training.py --overwrite
```

See [trace_tools/README.md](trace_tools/README.md) for the expected local file
layout and [docs/data_sources.md](docs/data_sources.md) for citations.

## Train from Scratch

Inspect the default internal trace recipe without starting training:

```bash
python agent/train.py --stage plan --run-id plan_check
```

Run the complete committed pipeline:

```bash
python agent/train.py --stage all --run-id local_selected_full
```

The committed recipes execute:

```text
collect -> sft -> offline_rl -> eval
```

Available recipes:

```text
agent/configs/train_local_selected_expert_window5_safety.yaml
agent/configs/train_external_mix_selected_expert_window5_safety.yaml
```

The two recipes write to independent checkpoint and training-statistics paths.
Do not mix a checkpoint from one trace distribution with return-to-go
statistics from another distribution.

## Results and Models

Each run is stored under:

```text
agent/runs/<run_id>/
```

Important files include:

```text
logs/train.log
manifest.json
eval/evaluation_index.json
eval/*.json
```

Training checkpoints may be synchronized to distribution-specific paths under
`agent/models/`. The online service only loads explicitly published fixed model
paths; see [docs/training.md](docs/training.md).

## Local Playback Demo

Start the processor, decision API, network simulator, and static web server:

```bash
python start_system.py
```

Open <http://127.0.0.1:3000/>. Upload an MP4 through the processor API, then use
the returned `video_id` in the web interface.

## Validation

```bash
python -m compileall -q agent processor simulator trace_tools start_system.py
python -m unittest discover -s tests -p "test_*.py"
```

## Cleanup

Remove caches and generated runs:

```powershell
.\scripts\clean_generated.ps1
```

Also remove generated model outputs:

```powershell
.\scripts\clean_generated.ps1 -IncludeModels
```

# Training and Evaluation

## Prerequisites

1. Install the Python dependencies.
2. Set `ABR_LLM_PATH` to a compatible local Hugging Face model directory.
3. Generate the required trace splits with the scripts in `trace_tools/`.

Trace data, base models, checkpoints, and run outputs are not distributed in
this repository.

## Inspect a Training Plan

The default recipe is the internal trace recipe:

```bash
python agent/train.py --stage plan --run-id plan_check
```

The external mix recipe can be inspected with:

```bash
python agent/train.py \
  --config agent/configs/train_external_mix_selected_expert_window5_safety.yaml \
  --stage plan \
  --run-id external_mix_plan
```

## Run from Scratch

```bash
python agent/train.py --stage all --run-id local_selected_full
```

The committed recipes execute:

```text
collect -> sft -> offline_rl -> eval
```

Use the same `run-id` when executing stages separately.

## Outputs

Each run is self-contained under:

```text
agent/runs/<run_id>/
  config.yaml
  manifest.json
  data/
  checkpoints/
  eval/
  logs/
```

The recipe may also synchronize checkpoints and matching training statistics to
distribution-specific paths under `agent/models/`.

## Publishing an Online Model

The online service searches these fixed paths:

```text
agent/models/netllm_rl/
agent/models/netllm_offline_rl/
agent/models/netllm_sft/
agent/models/training_stats.json
```

Publish a checkpoint together with the `training_stats.json` produced by the
same run. A checkpoint and return-to-go statistics from different trace
distributions are not compatible.

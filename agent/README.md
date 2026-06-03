# Agent

`agent/` contains the ABR simulation environment, policies, training pipeline,
evaluation tools, and online decision API.

## Main Entry Points

```bash
python agent/train.py --stage plan --run-id plan_check
python agent/train.py --stage all --run-id local_selected_full
python agent/evaluate.py --help
```

Available training stages:

```text
plan
collect
sft
offline_rl
rl
pensieve
eval
evaluate
all
```

## Committed Recipes

```text
configs/train_local_selected_expert_window5_safety.yaml
configs/train_external_mix_selected_expert_window5_safety.yaml
```

Both recipes start from expert collection and use independent checkpoint and
training-statistics paths. The local recipe uses internal traces; the external
mix recipe uses generated `external_mix_*` traces.

## Key Files

```text
abr_env.py              Offline ABR simulation environment
network.py              NetLLM-style ABR policy
data_collector.py       BOLA, MPC, and BBA expert collection
experience_dataset.py   SFT and offline RL window dataset
sft_trainer.py          Supervised behavior cloning
offline_rl_trainer.py   Return-conditioned offline RL
rl_trainer.py           Optional online RL fine-tuning
pensieve_torch.py       PyTorch Pensieve baseline
evaluate.py             Unified offline evaluation
train.py                Config-driven training orchestration
```

## Outputs

Runs are self-contained under `runs/<run_id>/`. Generated models are written
under `models/` and are ignored by Git.

The online decision service searches fixed model paths such as
`models/netllm_offline_rl/`. Publish a checkpoint together with the
`training_stats.json` from the same run and trace distribution.

See [`docs/training.md`](../docs/training.md) for details.

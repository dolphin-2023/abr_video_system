# Server training checklist

This is the recommended long-run entrypoint after the local smoke run.

## 1. Set the model path

Linux:

```bash
export ABR_LLM_PATH=/path/to/qwen3.5-4b-base
```

Windows PowerShell:

```powershell
$env:ABR_LLM_PATH="D:\ai-models\qwen3.5-4b-base"
```

`agent/configs/train_server.yaml` intentionally leaves `env.llm_path` empty so
the process-level `ABR_LLM_PATH` is respected on different machines.

## 2. Plan check

```bash
python agent/train.py --config agent/configs/train_server.yaml --stage plan --run-id server_plan
```

## 3. Full run

Prefer a stable run id so every stage can be resumed independently:

```bash
RUN_ID=server_full_$(date +%Y%m%d_%H%M)
python agent/train.py --config agent/configs/train_server.yaml --stage collect --run-id "$RUN_ID"
python agent/train.py --config agent/configs/train_server.yaml --stage sft --run-id "$RUN_ID"
python agent/train.py --config agent/configs/train_server.yaml --stage offline_rl --run-id "$RUN_ID"
python agent/train.py --config agent/configs/train_server.yaml --stage rl --run-id "$RUN_ID"
python agent/train.py --config agent/configs/train_server.yaml --stage pensieve --run-id "$RUN_ID"
python agent/train.py --config agent/configs/train_server.yaml --stage eval --run-id "$RUN_ID"
```

The staged form is safer than one huge `--stage all` command because failed or
interrupted stages can be resumed without regenerating earlier artifacts.

## 4. Useful quick commands

Compile and tests:

```bash
python -m compileall -q agent origin_traces processor simulator start_system.py
python -m unittest discover -s tests -p "test_*.py"
```

Evaluate only NetLLM models on external sets after training:

```bash
python agent/evaluate.py --trace-split external_puffer_recent --episodes 300 --sample-mode stratified --policies netllm-sft netllm-offline-rl netllm-rl --stats-path agent/runs/$RUN_ID/data/training_stats.json --sft-model-path agent/runs/$RUN_ID/checkpoints/sft --offline-rl-model-path agent/runs/$RUN_ID/checkpoints/offline_rl --rl-model-path agent/runs/$RUN_ID/checkpoints/rl --output agent/runs/$RUN_ID/eval/external_puffer_recent_netllm.json
python agent/evaluate.py --trace-split external_weak_mobile --episodes 252 --sample-mode stratified --policies netllm-sft netllm-offline-rl netllm-rl --stats-path agent/runs/$RUN_ID/data/training_stats.json --sft-model-path agent/runs/$RUN_ID/checkpoints/sft --offline-rl-model-path agent/runs/$RUN_ID/checkpoints/offline_rl --rl-model-path agent/runs/$RUN_ID/checkpoints/rl --output agent/runs/$RUN_ID/eval/external_weak_mobile_netllm.json
```

## 5. What changed in the server config

- More expert data: 320 general trajectories and 240 high-bandwidth trajectories.
- High-MPC is more willing to climb to top bitrates when buffer and estimated
  throughput are safe.
- SFT and offline RL train longer with slightly stronger class balancing.
- Online RL uses a smaller learning rate, KL anchoring, more validation traces,
  batched rollouts, and a less brittle safety gate.
- Final evaluation includes internal test, recent Puffer, and weak mobile splits.

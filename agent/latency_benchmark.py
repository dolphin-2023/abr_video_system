import argparse
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import torch
import uvicorn


DASH_SEGMENT_MS = 4000.0
DEFAULT_BITRATES_BPS = [300_000, 600_000, 900_000, 2_000_000, 4_500_000, 9_000_000]


def sync_device(device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def summarize(values_ms):
    values = [float(value) for value in values_ms]
    values_sorted = sorted(values)
    if not values_sorted:
        return {}

    def percentile(pct):
        if len(values_sorted) == 1:
            return values_sorted[0]
        index = (len(values_sorted) - 1) * (float(pct) / 100.0)
        lower = int(index)
        upper = min(lower + 1, len(values_sorted) - 1)
        weight = index - lower
        return values_sorted[lower] * (1.0 - weight) + values_sorted[upper] * weight

    mean_ms = statistics.fmean(values_sorted)
    return {
        "runs": len(values_sorted),
        "mean_ms": mean_ms,
        "max_ms": max(values_sorted),
        "min_ms": min(values_sorted),
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "std_ms": statistics.pstdev(values_sorted) if len(values_sorted) > 1 else 0.0,
        "dash_segment_ms": DASH_SEGMENT_MS,
        "mean_percent_of_4s_segment": mean_ms / DASH_SEGMENT_MS * 100.0,
        "max_percent_of_4s_segment": max(values_sorted) / DASH_SEGMENT_MS * 100.0,
        "mean_headroom_vs_4s": DASH_SEGMENT_MS / max(mean_ms, 1e-9),
    }


def build_direct_state(agent_module):
    session = agent_module.SessionState("latency-direct")
    return session.update(
        throughput_kbps=3500.0,
        buffer_level=12.0,
        quality_idx=2,
        download_time_ms=1200.0,
        available_bitrates_bps=DEFAULT_BITRATES_BPS,
        segment_duration=4.0,
    )


def benchmark_direct(agent_module, runs, warmup, episode_length):
    model = agent_module.netllm_agent
    device = agent_module.DEVICE
    state_matrix = build_direct_state(agent_module)
    state_tensor = (
        torch.tensor(state_matrix, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    target_return = float(agent_module.TARGET_RETURN)
    latencies = []

    model.eval()
    total = int(warmup) + int(runs)
    with torch.inference_mode():
        for step in range(total):
            if step % int(episode_length) == 0:
                history = model.new_history()
                target_return = float(agent_module.TARGET_RETURN)
            timestep = step % int(episode_length)
            sync_device(device)
            start = time.perf_counter()
            with agent_module.autocast_context():
                logits, return_emb, state_emb, time_emb = model.predict_logits(
                    state=state_tensor,
                    target_return=target_return,
                    timestep=timestep,
                    history=history,
                )
                action, _ = model.select_action(logits, deterministic=True)
            sync_device(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            model.append_history(
                history=history,
                return_emb=return_emb,
                state_emb=state_emb,
                time_emb=time_emb,
                action=action,
                device=device,
            )
            if step >= int(warmup):
                latencies.append(elapsed_ms)

    return latencies


def wait_for_server(base_url, timeout=180.0):
    deadline = time.time() + float(timeout)
    last_error = None
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return response.json()
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def start_inprocess_server(agent_module, host, port):
    config = uvicorn.Config(
        app=agent_module.app,
        host=host,
        port=int(port),
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="latency-uvicorn", daemon=True)
    thread.start()
    return server, thread


def benchmark_http(base_url, runs, warmup, episode_length):
    video_id = f"latency-http-{int(time.time())}"
    payload = {
        "video_id": video_id,
        "buffer_level": 12.0,
        "throughput_kbps": 3500.0,
        "available_bitrates": DEFAULT_BITRATES_BPS,
        "current_quality_index": 2,
        "segment_duration": 4.0,
        "last_reward": 1.0,
    }
    latencies = []

    with httpx.Client(timeout=None) as client:
        try:
            client.post(f"{base_url}/reset_session", json={"video_id": video_id})
        except Exception:
            pass
        total = int(warmup) + int(runs)
        for step in range(total):
            if step % int(episode_length) == 0:
                try:
                    client.post(f"{base_url}/reset_session", json={"video_id": video_id})
                except Exception:
                    pass
                payload["current_quality_index"] = 2
            start = time.perf_counter()
            response = client.post(f"{base_url}/abr_decision", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            result = response.json()
            payload["current_quality_index"] = int(result.get("quality_index", payload["current_quality_index"]))
            if step >= int(warmup):
                latencies.append(elapsed_ms)

    return latencies


def default_output_path():
    run_id = datetime.now().strftime("latency_benchmark_%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "runs" / run_id / "latency_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ABR-LLM inference latency.")
    parser.add_argument("--runs", type=int, default=200, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations excluded from stats.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=60,
        help="Reset simulated playback every N decisions; the DASH demo uses about 60 four-second chunks.",
    )
    parser.add_argument("--skip-direct", action="store_true", help="Skip direct model latency.")
    parser.add_argument("--skip-http", action="store_true", help="Skip /abr_decision HTTP latency.")
    parser.add_argument(
        "--http-url",
        default=None,
        help="Use an existing agent server, e.g. http://127.0.0.1:8081. If omitted, start an in-process server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_direct and args.skip_http:
        raise SystemExit("Nothing to benchmark: both --skip-direct and --skip-http were set.")

    import agent as agent_module

    output = Path(args.output) if args.output else default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "episode_length": int(args.episode_length),
        "dash_segment_ms": DASH_SEGMENT_MS,
        "environment": {
            "device": str(agent_module.DEVICE),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(agent_module.DEVICE)
                if getattr(agent_module.DEVICE, "type", None) == "cuda"
                else None
            ),
            "base_model": str(agent_module.BASE_MODEL_PATH),
            "active_policy": agent_module.ACTIVE_POLICY_NAME,
            "active_policy_path": str(agent_module.ACTIVE_POLICY_PATH)
            if agent_module.ACTIVE_POLICY_PATH
            else None,
            "target_return": float(agent_module.TARGET_RETURN),
            "rtg": agent_module.RTG_PROCESSOR.describe(),
        },
        "metrics": {},
    }

    if not args.skip_direct:
        print(f"[Latency] direct model benchmark: runs={args.runs} warmup={args.warmup}")
        direct = benchmark_direct(agent_module, args.runs, args.warmup, args.episode_length)
        payload["metrics"]["direct_model"] = summarize(direct)

    server = None
    thread = None
    if not args.skip_http:
        if args.http_url:
            base_url = args.http_url.rstrip("/")
            print(f"[Latency] HTTP benchmark using existing server: {base_url}")
            health = wait_for_server(base_url)
        else:
            base_url = f"http://{args.host}:{int(args.port)}"
            print(f"[Latency] starting in-process server: {base_url}")
            server, thread = start_inprocess_server(agent_module, args.host, args.port)
            health = wait_for_server(base_url)
        payload["http_health"] = health
        print(f"[Latency] HTTP /abr_decision benchmark: runs={args.runs} warmup={args.warmup}")
        http_latencies = benchmark_http(base_url, args.runs, args.warmup, args.episode_length)
        payload["metrics"]["http_abr_decision"] = summarize(http_latencies)

    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Latency] saved -> {output}")
    for name, metrics in payload["metrics"].items():
        print(
            f"[Latency] {name}: mean={metrics['mean_ms']:.2f}ms "
            f"max={metrics['max_ms']:.2f}ms "
            f"p95={metrics['p95_ms']:.2f}ms "
            f"mean/4s={metrics['mean_percent_of_4s_segment']:.3f}%"
        )

    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=5.0)


if __name__ == "__main__":
    main()

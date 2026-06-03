import asyncio
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


PROCESSOR_URL = "http://127.0.0.1:8080/video"
BASE_DIR = Path(__file__).resolve().parent
TRACE_ROOT = BASE_DIR / "traces" / "real_world_split"


def resolve_trace_dir(split: str | None = None) -> tuple[Path, str]:
    configured_dir = os.environ.get("ABR_SIM_TRACE_DIR")
    if configured_dir:
        return Path(configured_dir).resolve(), "custom"

    split_name = (split or os.environ.get("ABR_SIM_TRACE_SPLIT", "test")).strip().lower()
    if split_name not in {"train", "test"}:
        split_name = "test"

    return (TRACE_ROOT / split_name).resolve(), split_name


http_client: httpx.AsyncClient | None = None


class GlobalTokenBucket:
    """Shared token bucket used to throttle all DASH segment requests."""

    def __init__(self):
        self.bandwidth_trace = []
        self.current_index = 0
        self.current_bps = 5000 * 1000 / 8
        self.current_trace_name = "None"
        self.current_trace_dir, self.current_split = resolve_trace_dir()
        self.tokens = 0.0
        self.last_update_time = time.perf_counter()
        self.lock = asyncio.Lock()

    def load_trace(self, filepath: Path):
        values = []
        try:
            with filepath.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        values.append(float(line))
        except Exception as exc:
            print(f"[NetworkSim] failed to load trace {filepath}: {exc}")
            return False

        if not values:
            return False

        self.bandwidth_trace = values
        self.current_index = 0
        self.current_trace_name = filepath.name
        print(
            f"[NetworkSim] loaded trace={self.current_trace_name} "
            f"seconds={len(self.bandwidth_trace)}"
        )
        return True

    def load_random_trace(self, split: str | None = None):
        trace_dir, split_name = resolve_trace_dir(split)
        if not trace_dir.exists():
            return False

        trace_files = [path for path in trace_dir.iterdir() if path.is_file()]
        if not trace_files:
            return False

        success = self.load_trace(random.choice(trace_files))
        if success:
            self.current_trace_dir = trace_dir
            self.current_split = split_name
        return success

    async def run_environment_loop(self):
        while True:
            if self.bandwidth_trace:
                kbps = self.bandwidth_trace[self.current_index]
                self.current_bps = max(1.0, kbps * 1000 / 8)

                if self.current_index % 10 == 0:
                    print(
                        f"[NetworkSim] trace={self.current_trace_name} "
                        f"t={self.current_index}s bandwidth={kbps:.1f} Kbps"
                    )

                self.current_index = (self.current_index + 1) % len(self.bandwidth_trace)
            await asyncio.sleep(1.0)

    async def consume(self, amount: int) -> None:
        remaining = int(amount)
        while remaining > 0:
            async with self.lock:
                now = time.perf_counter()
                elapsed = now - self.last_update_time
                self.last_update_time = now

                self.tokens += elapsed * self.current_bps
                burst_limit = self.current_bps * 0.05
                if self.tokens > burst_limit:
                    self.tokens = burst_limit

                if self.tokens > 0:
                    take = min(remaining, self.tokens)
                    self.tokens -= take
                    remaining -= take

            if remaining > 0:
                await asyncio.sleep(0.01)


bw_bucket = GlobalTokenBucket()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=None)
    bw_bucket.load_random_trace()
    loop_task = asyncio.create_task(bw_bucket.run_environment_loop())
    try:
        yield
    finally:
        loop_task.cancel()
        if http_client:
            await http_client.aclose()


app = FastAPI(title="ABR Network Simulator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/sim/random_trace")
async def api_random_trace(split: str | None = None):
    success = bw_bucket.load_random_trace(split=split)
    if success:
        return JSONResponse(
            {
                "status": "success",
                "split": bw_bucket.current_split,
                "trace": bw_bucket.current_trace_name,
            }
        )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "trace 加载失败"},
    )


@app.get("/api/sim/status")
async def api_sim_status():
    current_kbps = (bw_bucket.current_bps * 8) / 1000 if bw_bucket.current_bps else 0
    return JSONResponse(
        {
            "trace_name": bw_bucket.current_trace_name,
            "trace_split": bw_bucket.current_split,
            "trace_dir": str(bw_bucket.current_trace_dir),
            "current_kbps": current_kbps,
        }
    )


@app.get("/video/{path:path}")
async def proxy_video(path: str, request: Request):
    if http_client is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "simulator is not ready"},
        )

    target_url = f"{PROCESSOR_URL}/{path}"
    headers = {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in request.headers.raw}
    headers.pop("host", None)

    upstream_request = http_client.build_request("GET", target_url, headers=headers)
    upstream_response = await http_client.send(upstream_request, stream=True)

    async def throttled_stream():
        try:
            async for chunk in upstream_response.aiter_bytes(8192):
                await bw_bucket.consume(len(chunk))
                yield chunk
        except asyncio.CancelledError:
            pass
        finally:
            await upstream_response.aclose()

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() != "transfer-encoding"
    }
    response_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response_headers["Pragma"] = "no-cache"
    response_headers["Expires"] = "0"

    return StreamingResponse(
        throttled_stream(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy:app", host="127.0.0.1", port=8082, reload=True)

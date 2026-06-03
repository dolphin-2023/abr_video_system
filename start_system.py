import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_HOST = "127.0.0.1"


@dataclass(frozen=True)
class Service:
    name: str
    command: list[str]
    cwd: Path
    url: str


SERVICES = [
    Service(
        name="processor",
        command=[
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            LOCAL_HOST,
            "--port",
            "8080",
        ],
        cwd=ROOT_DIR / "processor",
        url=f"http://{LOCAL_HOST}:8080",
    ),
    Service(
        name="agent",
        command=[
            sys.executable,
            "-m",
            "uvicorn",
            "agent:app",
            "--host",
            LOCAL_HOST,
            "--port",
            "8081",
        ],
        cwd=ROOT_DIR / "agent",
        url=f"http://{LOCAL_HOST}:8081",
    ),
    Service(
        name="simulator",
        command=[
            sys.executable,
            "-m",
            "uvicorn",
            "proxy:app",
            "--host",
            LOCAL_HOST,
            "--port",
            "8082",
        ],
        cwd=ROOT_DIR / "simulator",
        url=f"http://{LOCAL_HOST}:8082",
    ),
    Service(
        name="web",
        command=[
            sys.executable,
            "-m",
            "http.server",
            "3000",
            "--bind",
            LOCAL_HOST,
        ],
        cwd=ROOT_DIR / "web",
        url=f"http://{LOCAL_HOST}:3000",
    ),
]


def start_services() -> list[tuple[Service, subprocess.Popen]]:
    processes = []
    print("[System] Starting ABR video system services...")

    for service in SERVICES:
        print(f"[System] Starting {service.name} in {service.cwd}")
        process = subprocess.Popen(service.command, cwd=str(service.cwd))
        processes.append((service, process))

    print("\n[System] All services have been started.")
    for service, process in processes:
        print(f"[System] {service.name:<9} pid={process.pid:<6} {service.url}")
    print(f"[System] Open {SERVICES[-1].url}/")
    print("[System] Press Ctrl+C to stop all services.\n")
    return processes


def stop_services(processes: list[tuple[Service, subprocess.Popen]]) -> None:
    print("\n[System] Stopping services...")

    for service, process in reversed(processes):
        if process.poll() is None:
            print(f"[System] Stopping {service.name}")
            process.terminate()

    for service, process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[System] Killing {service.name}")
            process.kill()

    print("[System] Shutdown complete.")


def main() -> None:
    processes = start_services()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_services(processes)


if __name__ == "__main__":
    main()

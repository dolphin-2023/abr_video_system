# Web

`web/` contains the browser playback page. It displays video, buffer,
throughput, and QoE metrics while requesting bitrate decisions from the agent.

## Service

```bash
python -m http.server 3000 --directory web
```

Open <http://127.0.0.1:3000/>.

The page expects:

```text
processor  http://127.0.0.1:8080
agent      http://127.0.0.1:8081
simulator  http://127.0.0.1:8082
```

Run `python start_system.py` from the repository root to start all local
services.

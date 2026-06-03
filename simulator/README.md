# Simulator

`simulator/` is a trace-driven HTTP proxy. It fetches DASH files from the
processor service and throttles segment responses according to a local network
trace.

## Service

```bash
python -m uvicorn proxy:app --app-dir simulator --host 127.0.0.1 --port 8082
```

The default trace split is `test` under
`simulator/traces/real_world_split/test/`.

Override the split or use a custom trace directory:

```powershell
$env:ABR_SIM_TRACE_SPLIT = "train"
$env:ABR_SIM_TRACE_DIR = "D:\path\to\traces"
```

Generate traces with the scripts in [`trace_tools/`](../trace_tools/).

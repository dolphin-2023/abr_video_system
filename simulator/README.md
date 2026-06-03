# Simulator 网络模拟器

`simulator/` 是一个 trace 驱动的 HTTP 代理。它从 Processor 获取 DASH 文件，并根据本地网络 trace 对每个分片响应进行限速，模拟真实网络环境下的视频传输。

## 启动服务

```bash
python -m uvicorn proxy:app --app-dir simulator --host 127.0.0.1 --port 8082
```

默认使用 `simulator/traces/real_world_split/test/` 下的测试集 trace。

覆盖 split 或使用自定义 trace 目录：

```powershell
$env:ABR_SIM_TRACE_SPLIT = "train"
$env:ABR_SIM_TRACE_DIR = "D:\path\to\traces"
```

使用 [`trace_tools/`](../trace_tools/) 中的脚本生成 trace 文件。

---

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

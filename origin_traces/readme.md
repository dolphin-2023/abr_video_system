# origin_traces

这里存放原始网络 trace 数据，以及把外部数据转换成 ABR 仿真 trace 的工具。

## 目录

```text
new/                                新下载的外部测试数据
Real-world-bandwidth-traces-master/ 经典训练/测试 trace 来源
import_external_datasets.py         外部测试集导入工具
process_puffer.py                   旧 Puffer 处理脚本
process_traces.py                   旧 trace 整理脚本
```

## 新外部测试集导入

重新生成外部测试 split：

```powershell
D:\Software\miniforge3\envs\myenv\python.exe origin_traces\import_external_datasets.py --clear --max-traces 300 --min-seconds 90
```

默认输入：

```text
origin_traces/new/Puffer/
origin_traces/new/A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements/
origin_traces/new/Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics/
```

默认输出：

```text
simulator/traces/external_puffer_recent/
simulator/traces/external_weak_mobile/
```

处理策略：

```text
Puffer: 按 (session_id, index, channel) 分流；delivery_rate 从 bytes/s 转 Kbps；按中位吞吐分桶抽样。
PERFORM: 主动测速下行字段按 Kbps 处理，并按 Campaign/Operator/Scenario/RAT Info 分组。
LTE: 只保留 State == D 且 DL_bitrate > 0 的连续下载段。
```

生成后的 split 只用于外部测试，不参与训练。

# Trace 数据准备工具

Trace 数据没有提交到本仓库。请参阅 [`docs/data_sources.md`](../docs/data_sources.md)
了解数据来源归属及数据集特定说明。

## 本地数据布局

将下载或克隆的原始数据集放入以下目录结构：

```text
data/raw/
  Real-world-bandwidth-traces/
  Puffer/
  A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements/
  Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics/
```

生成的 trace 文件写入 `simulator/traces/` 目录。

## 内部数据集的训练/测试集划分

克隆内部数据源：

```bash
git clone https://github.com/confiwent/Real-world-bandwidth-traces \
  data/raw/Real-world-bandwidth-traces
```

生成确定性的内部训练/测试集：

```bash
python trace_tools/prepare_internal_traces.py --overwrite
```

输出结果：

```text
simulator/traces/real_world_split/train/
simulator/traces/real_world_split/test/
simulator/traces/real_world_split/manifest.json
```

## 外部数据源池

按上述布局将下载的外部数据放入 `data/raw/`，然后执行：

```bash
python trace_tools/import_external_datasets.py --clear --max-traces 300 --min-seconds 90
```

这会在以下目录生成外部 trace：

```text
simulator/traces/external_puffer_recent/
simulator/traces/external_weak_mobile/
```

本项目使用的 Puffer 数据源是 `2026-05-06 11:00:00 UTC` 至 `2026-05-07 11:00:00 UTC`
时间范围内的 `video_sent` 日志。

## 外部混合训练集划分

从外部数据源池创建确定性的训练、验证和测试集：

```bash
python trace_tools/split_external_for_training.py --overwrite
```

这会生成 `external_mix_train`、`external_mix_valid` 和 `external_mix_test` 三个
split 目录。

---

# Trace Preparation Tools

Trace data is not committed to this repository. See
[`docs/data_sources.md`](../docs/data_sources.md) for source attribution and
dataset-specific notes.

## Local Data Layout

Download or clone the raw datasets under:

```text
data/raw/
  Real-world-bandwidth-traces/
  Puffer/
  A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements/
  Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics/
```

Generated traces are written under `simulator/traces/`.

## Internal Train/Test Splits

Clone the internal source dataset:

```bash
git clone https://github.com/confiwent/Real-world-bandwidth-traces \
  data/raw/Real-world-bandwidth-traces
```

Generate deterministic internal splits:

```bash
python trace_tools/prepare_internal_traces.py --overwrite
```

The output is:

```text
simulator/traces/real_world_split/train/
simulator/traces/real_world_split/test/
simulator/traces/real_world_split/manifest.json
```

## External Source Pools

Place the downloaded external files in `data/raw/` using the layout above, then
run:

```bash
python trace_tools/import_external_datasets.py --clear --max-traces 300 --min-seconds 90
```

This creates:

```text
simulator/traces/external_puffer_recent/
simulator/traces/external_weak_mobile/
```

The Puffer source used in this project is the `video_sent` log from
`2026-05-06 11:00:00 UTC` to `2026-05-07 11:00:00 UTC`.

## External Mix Splits

To create deterministic train, validation, and test splits from the external
source pools:

```bash
python trace_tools/split_external_for_training.py --overwrite
```

This creates `external_mix_train`, `external_mix_valid`, and
`external_mix_test`.

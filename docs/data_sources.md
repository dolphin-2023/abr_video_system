# 网络 Trace 数据来源

本仓库不分发原始或处理后的网络 trace 数据。使用者须从各数据集的原始发布方获取数据，
查阅其许可证或使用条款，并在训练或评估前生成本地 ABR trace 划分。

## 内部数据集

内部训练和测试 trace 来源于：

- confiwent, **Real-world-bandwidth-traces**：
  <https://github.com/confiwent/Real-world-bandwidth-traces>

该集合包含多个网络场景下预处理后的真实吞吐量 trace，包括 3G/HSDPA、FCC、Oboe 以及
历史 Stanford Puffer 样本。本地准备流程将原始数值从 Mbit/s 转换为 1 Hz Kbit/s 的
ABR trace，并创建确定性的训练/测试集划分。

## 外部数据集

外部数据集与内部数据集保持分离，以便独立研究分布偏移（distribution shift）和策略鲁棒性。

### Stanford Puffer

- Stanford Puffer, **Experiment Results**：
  <https://puffer.stanford.edu/results/>

选用的数据源为以下 UTC 时间范围内的 `video_sent` 日志：

```text
2026-05-06 11:00:00 UTC 至 2026-05-07 11:00:00 UTC
```

导入流程按 `(session_id, index, channel)` 分组记录，将 `delivery_rate` 从 bytes/s
转换为 Kbit/s，将每条流重采样为 1 Hz，并按中位吞吐量分层筛选 trace。

### 大规模 4G、NB-IoT 和 5G NSA 测量数据集

- Kousias, K., Rajiullah, M., Caso, G., 等. "A large-scale dataset of 4G,
  NB-IoT, and 5G non-standalone network measurements." *IEEE Communications
  Magazine*, 2024, 62(5): 44-49.
- 数据集 DOI: <https://doi.org/10.5281/zenodo.8224890>

导入流程读取活跃下行吞吐量测量数据，将其转换为弱网/移动网络环境下的 ABR trace。

### Beyond Throughput 4G LTE 数据集

- Raca, D., Quinlan, J., Zahran, A. H., 等. "Beyond throughput: a 4G LTE
  dataset with channel and context metrics." In *Proceedings of the 9th ACM
  Multimedia Systems Conference*, ACM, 2018, pp. 460-465.
- 数据集: <https://zenodo.org/records/1219679>

导入流程保留具有正向下行比特率的连续下载行，并将其转换为弱网/移动网络环境下的 ABR trace。

## 仓库数据策略

以下目录为本地数据位置，已被 Git 忽略：

```text
data/raw/
simulator/traces/
```

生成的 manifest 文件可以包含数据集统计信息和 split 归属关系，但在公开发布前
不得包含本机私有路径或原始数据集内容。

---

# Network Trace Data Sources

This repository does not distribute raw or processed network trace data. Users
must obtain each dataset from its original source, review the source license or
terms of use, and generate local ABR trace splits before training or evaluation.

## Internal Dataset

The internal training and test traces are derived from:

- confiwent, **Real-world-bandwidth-traces**:
  <https://github.com/confiwent/Real-world-bandwidth-traces>

That collection contains preprocessed real-world throughput traces from several
network scenarios, including 3G/HSDPA, FCC, Oboe, and historical Stanford
Puffer samples. The local preparation workflow converts the source values from
Mbit/s to 1 Hz Kbit/s ABR traces and creates deterministic train/test splits.

## External Datasets

The external datasets are kept separate from the internal dataset so that they
can be used to study distribution shift and robustness.

### Stanford Puffer

- Stanford Puffer, **Experiment Results**:
  <https://puffer.stanford.edu/results/>

The selected source is the `video_sent` log for the UTC interval:

```text
2026-05-06 11:00:00 UTC to 2026-05-07 11:00:00 UTC
```

The import workflow groups records by `(session_id, index, channel)`, converts
`delivery_rate` from bytes/s to Kbit/s, resamples each stream to 1 Hz, and
selects traces by median-throughput strata.

### Large-Scale 4G, NB-IoT, and 5G NSA Measurements

- Kousias, K., Rajiullah, M., Caso, G., et al. "A large-scale dataset of 4G,
  NB-IoT, and 5G non-standalone network measurements." *IEEE Communications
  Magazine*, 2024, 62(5): 44-49.
- Dataset DOI: <https://doi.org/10.5281/zenodo.8224890>

The import workflow reads active downlink throughput measurements and converts
them into weak/mobile-network ABR traces.

### Beyond Throughput 4G LTE Dataset

- Raca, D., Quinlan, J., Zahran, A. H., et al. "Beyond throughput: a 4G LTE
  dataset with channel and context metrics." In *Proceedings of the 9th ACM
  Multimedia Systems Conference*, ACM, 2018, pp. 460-465.
- Dataset: <https://zenodo.org/records/1219679>

The import workflow keeps continuous download rows with positive downlink
bitrate and converts them into weak/mobile-network ABR traces.

## Repository Policy

The following directories are local data locations and are intentionally
ignored by Git:

```text
data/raw/
simulator/traces/
```

Generated manifests may contain dataset statistics and split membership, but
must not contain private machine paths or raw dataset contents before
publication.

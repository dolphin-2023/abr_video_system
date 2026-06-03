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

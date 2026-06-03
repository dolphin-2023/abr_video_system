# Generated Trace Directory

This directory is populated by the scripts in [`trace_tools/`](../../trace_tools/).
Trace files and generated manifests are local artifacts and are intentionally
excluded from Git.

The simulator and training environment resolve split names from this directory,
for example:

```text
real_world_split/train/
real_world_split/test/
external_mix_train/
external_mix_valid/
external_mix_test/
```

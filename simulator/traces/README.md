# 生成的 Trace 目录

此目录由 [`trace_tools/`](../../trace_tools/) 中的脚本填充。
Trace 文件和生成的 manifest 属于本地产物，不会被提交到 Git。

模拟器和训练环境从此目录解析 split 名称，例如：

```text
real_world_split/train/
real_world_split/test/
external_mix_train/
external_mix_valid/
external_mix_test/
```

---

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

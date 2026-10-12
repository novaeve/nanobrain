# Dataset Artifact

The frozen 100k metadata-v2 bundle is stored at:

```text
assets/nanobrain_v3_100k_metadata_v2/
```

That directory is excluded from Git. The company-hosted archive URL is supplied through `NANOBRAIN_DATA_URL` (or `--url`) after the data release location is approved. `NANOBRAIN_DATA_SHA256` optionally pins the downloaded ZIP/TAR archive. The bundle itself must contain `manifest.json`, `train/`, `validation/`, and `test/` with the existing 60k/20k/20k row assignment.

Run `uv run nanobrain-data prepare` to download and validate it. For a local archive, use `uv run nanobrain-data prepare --archive /path/to/archive`. The command verifies all manifest-listed files and refuses an unexpected split, identity version, or train-normalization contract.

The bundle is not regenerated from the 42.8M-cell corpus during setup. This keeps the released 100k row selection and partition unchanged and avoids requiring access to the full internal corpus.

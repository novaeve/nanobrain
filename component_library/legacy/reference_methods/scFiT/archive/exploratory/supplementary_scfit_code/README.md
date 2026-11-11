# scFiT Supplementary Code

This archive contains the minimal code needed to reproduce the scFiT training and
evaluation pipeline used in the paper. It intentionally excludes plotting scripts,
baseline implementations, trained checkpoints, generated prediction artifacts, and
large data files.

## Contents

- `model.py`: scFiT Transformer model with target-state flow-matching conditioning.
- `utils.py`: shared training and Sinkhorn-regularized OT utilities.
- `route_a/`: scFiT temporal prediction training, evaluation, protocol loading, and
  evaluation-asset preparation code.
- `configs/`: YAML configs for the reported scFiT dataset-task protocols.
- `protocols/`: observed/held-out temporal split definitions.
- `scripts/prepare_route_a_assets.py`: prepares PCA/kNN/marker evaluation assets from
  a protocol and preprocessed bundle.
- `scripts/evaluate_route_a_artifact.py`: re-evaluates saved prediction artifacts.

## What Is Not Included

- No baseline method code is included.
- No plotting or paper-figure scripts are included.
- No model weights, checkpoints, generated artifacts, or logs are included.
- No raw or preprocessed datasets are included. The paper uses public datasets cited in
  the manuscript.

## Expected Data Layout

Each protocol uses a `<PREPROCESSED_BUNDLE_DIR>` placeholder. Replace it with the path to
a preprocessed bundle containing:

```text
<PREPROCESSED_BUNDLE_DIR>/
  dataset_metadata.json
  cell_metadata.pkl
  expression_log1p_clipped.npy
  obs_names.npy
  hvg_gene_indices.npy
  hvg_var_names.npy
  hvg_gene_names.npy
```

The metadata must contain `author_day` and `author_day_numeric`. Mouse neural crest
bundles may additionally contain `author_cell_type` for the auxiliary biological metrics.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The experiments in the paper were run with CUDA-enabled PyTorch on single NVIDIA H100
GPUs. The code also runs on CPU for debugging, but full training is intended for GPU use.

## Running scFiT

Before running, replace the placeholders in the selected config and protocol file:

```text
<RAW_DATA_FILE>
<PREPROCESSED_BUNDLE_DIR>
<PROTOCOL_MANIFEST_PATH>
<EXPERIMENT_OUTPUT_DIR>
<EVALUATION_ASSETS_DIR>
```

From the archive root, train a mouse neural crest near-extrapolation model after filling
these placeholders with local paths:

```bash
python -m route_a.main --config configs/mouse/config_mouse_route_a_near.yaml
```

Other reported scFiT configs are available under:

```text
configs/mouse/
configs/benchmarks/
```

Evaluation assets are prepared automatically during training if they are missing. They
can also be prepared explicitly:

```bash
python scripts/prepare_route_a_assets.py \
  --protocol <PROTOCOL_MANIFEST_PATH> \
  --output-dir <EVALUATION_ASSETS_DIR>
```

To re-evaluate a saved prediction artifact:

```bash
python scripts/evaluate_route_a_artifact.py \
  --artifact-dir <PREDICTION_ARTIFACT_DIR> \
  --protocol <PROTOCOL_MANIFEST_PATH> \
  --assets-dir <EVALUATION_ASSETS_DIR>
```

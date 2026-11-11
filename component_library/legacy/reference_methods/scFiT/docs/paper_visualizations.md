# Standalone visualization scripts for Route A paper figures

This folder contains new plotting code only. It does not modify the existing training, evaluation, or baseline pipelines.

## Script

- `plot_mouse_neural_crest_umap_pca.py`
- `plot_mouse_neural_crest_compare.py`

It generates UMAP and/or PCA panels for the `mouse_neural_crest` dataset for the three Route A tasks:

- interpolation
- near extrapolation
- far extrapolation

`plot_mouse_neural_crest_compare.py` is the paper-facing combined figure script for
Figure 3. It renders a 3x3 layout:

1. held-out target populations
2. scFiT predictions
3. scNODE predictions

for interpolation, near extrapolation, and far extrapolation. The design follows
the revised paper narrative:

- observed training context is shown in light gray
- source timepoint(s) used for generation are shown in dark gray
- only the target timepoints being compared are shown in color

This avoids the older "all true timepoints / observed timepoints" layout that was
harder to interpret.

Each figure contains four unlabeled panels in fixed order:

1. all true timepoints
2. after removal
3. scFiT predictions
4. strongest baseline predictions

The output images do **not** include figure titles or panel titles, so they can be placed directly into the paper layout.
Current strongest baseline is set to `scNODE`, based on the existing Route A metrics in this repository.

## Recommended run

```bash
python results/figures/plot_mouse_neural_crest_umap_pca.py \
  --tasks interpolation near far \
  --embedding both \
  --max-fit-cells 15000 \
  --max-plot-cells-per-timepoint 800 \
  --output-dir results/figures/umap_pca_panels
```

## Progress output

The script now prints both a `tqdm` completion bar and step-by-step status logs, for example:

- current job index like `[1/6]`
- current task and embedding
- panel cell counts
- embedding fit / transform stages
- saved output path

## Why it was hanging before

The first version tried to fit/transform UMAP directly on very large 4096-dimensional matrices, which made it look stuck for a long time.
The updated version reduces plotting load by subsampling cells per timepoint and fitting UMAP on PCA-50 features first, which is much more practical for this dataset.

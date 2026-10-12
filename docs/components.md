# nanoBrain Component API

## Fixed scope

The public entry point uses the fixed metadata-v2 100k dataset at `assets/nanobrain_v3_100k_metadata_v2/` and four independently trained tasks: annotation, integration, development, and spatial niche. The normal training path opens train and validation only. The test partition and its published leaderboard are historical descriptive evaluation artifacts.

## Model components

`src/nanobrain/registry.py` registers factories for `encoder`, `attention`, `spatial`, `residual`, `pairing`, `flow`, and `head`. A `CandidateGenome` selects a compatible factory at each position. The default model combines an MLP expression encoder, differential linear attention, bounded neighbor-mean fusion, N-BEATS residual correction, Sinkhorn matching, and a Flow Matching Transformer.

| Component | Input / output contract | Built-in default |
|---|---|---|
| Encoder | `[B,G]` to `[B,H]` or `[B,T,H]` | MLP |
| Attention | Preserve encoder shape | Differential linear |
| Spatial | Hidden states plus bounded within-split neighbors | Neighbor mean |
| Residual | Preserve hidden width | N-BEATS-style correction |
| Pairing | Source / target populations to transport plan | Sinkhorn |
| Flow | Source state and condition to population velocity | FMT |
| Task head | Hidden state to task-specific prediction | Classification or regression |

Only eligible rows contribute to a task objective. Integration uses expression reconstruction and a gradient-reversal batch classifier; it does not use cell-type or niche labels as supervision. Development uses explicit unpaired cross-sectional source/target populations and a condition. Spatial context is limited to independently verified within-split sections; unavailable geometry contributes no cross-cell neighbor. The current niche label is an anatomy-derived proxy, not independently curated niche ground truth.

## Customization

See [Train Your Own Model Component](docs/custom-model.md) for a complete encoder example. `--plugin MODULE` explicitly imports the component module before the genome is loaded. The run receipt stores module names and source hashes. Registering a plugin expands explicit component selection; it does not change the frozen built-in mutation search space.

## Optional research components

The copied research code also contains optional upstream component adapters and graph-based architectures. They require their own approved upstream code or packages and are not installed by the CPU quick start. The public quick start makes no claim that every optional reference method is bundled or that its pretrained assets are redistributable.

## Data contract

The 60k/20k/20k row assignment and 256-gene panel stay frozen. Train-only normalization and manifest hashes are validated during preparation. Metadata-v2 repairs row identity and retains only independently verified geometry. The geometry limitation must be considered when interpreting spatial metrics, particularly because test geometry does not have equivalent independent verification.

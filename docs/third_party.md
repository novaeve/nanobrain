# Third-Party Components

The CPU quick start depends on PyTorch, NumPy, SciPy, pandas, PyArrow, scikit-learn, POT, and PyYAML. Exact resolved versions and sources are recorded in `uv.lock`. Optional benchmark extras add AnnData, Scanpy, and scib-metrics. Installation of a dependency does not transfer its license to nanoBrain.

## Included shared source

The `src/brainwm/`, `src/brainwm_data/`, and `src/brainwm_benchmark/` packages are shared project components retained for the nanoBrain trainer and evaluator. The default nanoBrain route uses the project-native data and task contracts, PyTorch operators, and the existing FMT/transport implementation. The small public model is not a packaged deployment of the full Brain World Model.

## Optional upstream adapters

The source tree retains adapters for external methods and alternative components. The raw `reference_methods/` snapshots, pretrained checkpoints, and model assets are excluded from public Git. They must be supplied by users from their respective approved upstream releases, with each upstream license and provenance preserved. The default CPU setup does not install or download these optional model assets.

Historical benchmark methods include BBKNN, CellTypist, Geneformer, GraphST, Harmony, Scanorama, scFiT, scGPT, scNODE, scVI/scANVI, TrajectoryNet, and XGBoost. Their checked-in scores are evaluation records, not a promise that every upstream training stack is bundled in the basic environment. Inspect each adapter's runtime requirements before selecting it.

## Optional kernel source

`src/flash_bla/` is a separate vendored GPU kernel snapshot. Its own `LICENSE` and `UPSTREAM.md` must accompany it if distributed. It is not part of the default nanoBrain CPU path. Other optional acceleration backends are installed only by explicit user choice.

## Release metadata

The company will specify the project's code license, citation metadata, and approved data-host URL separately. The dataset license and upstream model-asset licenses must be recorded independently from the project's code license when the release is published.

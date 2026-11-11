# scripts

Command-line entrypoints for smoke tests, training, evaluation, asset preparation, baseline runs, and figure generation.

- `smoke_test.py`: synthetic end-to-end pipeline check.
- `train.py`: maintained scFiT Route A training entrypoint.
- `evaluate.py`: maintained prediction artifact evaluation entrypoint.
- `prepare_route_a_assets.py`: build PCA, kNN, and marker-gene evaluation assets.
- `prepare_route_a_benchmark_bundle.py`: convert supported benchmark CSV inputs into processed Route A bundles.
- `run_*_protocol.py`: one-off baseline wrappers.
- `run_*.sh`: batch scripts for reproduced experiment sweeps.
- `figures/`: scripts for reproducing paper figures from generated artifacts.

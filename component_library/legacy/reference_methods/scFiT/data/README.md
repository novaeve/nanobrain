# data

Local-only storage for raw and processed datasets; no dataset files are included in this repository.

Expected layout:

- `data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad`
- `data/processed/neural_crest_pns_glia_hvg4096/`
- `data/raw/benchmarks/`
- `data/processed/benchmarks/`

Preprocessing scripts live in `scripts/prepare_hvg_panel.py` and `scripts/prepare_route_a_benchmark_bundle.py`.

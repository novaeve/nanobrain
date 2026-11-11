# Code Manifest

## Core scBrain

- `src/models/mjm.py`: SEA-AD three-level scBrain and ablation variants.
- `src/models/mjm_nlevel.py`: N-level scBrain used by Kukanja MS/EAE.
- `src/models/scbrain.py`: public scBrain aliases that wrap the legacy `mjm_*` classes.
- `src/engines/trainer.py`: SEA-AD training/evaluation engine.
- `src/engines/trainer_kukanja.py`: N-level Kukanja training/evaluation engine.
- `src/utils/dataloader.py`: SEA-AD packed NPZ loader and donor split.
- `src/utils/dataloader_kukanja.py`: Kukanja packed NPZ loader and sample split.

## Main Entry Points

- `experiments/scbrain/main.py`: public SEA-AD scBrain entry point.
- `experiments/kukanja/main_scbrain.py`: public Kukanja MS/EAE scBrain entry point.
- `experiments/mjm/main.py` and `experiments/kukanja/main_mjm1.py`: legacy-compatible entry points retained for reproducing historical run ids.
- `experiments/sklearn_baselines/main.py`: classic baselines.
- `experiments/Geneformer/main.py`, `experiments/scGPT/main.py`, `experiments/UCE/main.py`: major foundation baselines.

## Result Utilities

- `experiments/extract_all_metrics.py`: parse all included record logs.
- `experiments/kukanja/collect_results.py`: parse Kukanja logs.
- `scripts/collect_results.py`: legacy SEA-AD summary helper.
- `paper_tools/`: table and figure generation scripts used for the manuscript.
- `docs/RUNS.md`: paper result block to log/command mapping.
- `tools/check_release.py`: source-only release validation.
- `tools/make_archive.sh`: reproducible tarball creation.

## Exclusions

The following are intentionally excluded:

- raw datasets and packed `.npz` files
- pretrained foundation-model weights
- trained checkpoints
- extracted `test_features.npz`
- notebook scratch work
- local `__pycache__` files

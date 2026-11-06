# configs

YAML files for training runs, ablations, benchmark tasks, and protocol manifests.

- `mouse/`: maintained mouse neural crest Route A configs.
- `benchmark/`: maintained benchmark dataset configs.
- `ablation/mouse/`: ablation configs for the mouse Route A task.
- `protocols/mouse/`: held-out timepoint definitions for the mouse task.
- `protocols/benchmark/`: held-out timepoint definitions for benchmark tasks.

Most training hyperparameters live in the `training`, `validation`, `inference`, `ot`, and `evaluation` sections of these YAML files. Code defaults remain only as fallbacks for omitted optional fields.

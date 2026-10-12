# Train Your Own Model Component

The public component registry lets you replace an encoder while reusing the frozen data loader, task objectives, and trainer. Start with the small [`examples/custom_encoder.py`](../examples/custom_encoder.py) example.

The encoder receives expression features shaped `[batch, genes]` and returns a tensor shaped `[batch, hidden]` or `[batch, tokens, hidden]`. The configured hidden width must match the last dimension. It must preserve the cell axis; it may not mix unrelated cells.

## Register and run

The example registers the name `tutorial_encoder`. Its configuration lives in [`examples/custom_genome.json`](../examples/custom_genome.json). Import it with `--plugin` before constructing the genome:

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --plugin examples.custom_encoder --genome-json examples/custom_genome.json \
  --task annotation --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/custom-annotation
```

Replace the neural layers in `tutorial_encoder`, keep the input/output contract, and repeat the command with a new output directory. The other four-task targets use the same component API. Plugins are explicit imports; the trainer records each imported module name and source hash in its receipt. The fixed built-in mutation search space is not automatically expanded by a plugin.

Evaluate the completed run on validation with the same explicit plugin:

```bash
uv run python scripts/run_nanobrain_baselines.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --run-dir results/custom-annotation --plugin examples.custom_encoder \
  --tasks annotation --skip-baselines --evaluation-split validation \
  --device cpu --no-update-latest --output results/custom-annotation-eval
```

`--run-dir` reads the one completed training receipt and chooses its recorded best checkpoint (or final checkpoint when no best checkpoint is retained). It rejects incomplete or unstable runs. Evaluation produces prediction artifacts, metric receipts, and `benchmark_table.csv` using the same evaluator as the baseline runner.

## Compare fairly

Keep the same data manifest, task, seed, budget, and evaluation settings when comparing an altered component. Use train/validation while iterating. Treat comparisons with the published test leaderboard as descriptive only; test labels have already been opened.

The reference model definition is compact, but the entire training, data, and metric stack is larger than 500 lines. We do not claim a 500-line complete pipeline or a fixed time-to-train without a hardware-specific measurement.

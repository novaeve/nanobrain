# Unified Results Summary

This release keeps the lightweight experiment logs used by the paper. To avoid stale copied tables, use the parser as the source of truth:

```bash
python experiments/extract_all_metrics.py
```

The parser reports SEA-AD annotation metrics, missing-result status, batch-integration metrics, and Kukanja transfer-learning metrics from the included logs and CSV summaries.

For a manuscript-oriented map from result blocks to files, see `docs/RUNS.md`.

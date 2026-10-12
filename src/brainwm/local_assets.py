"""Resolve runtime assets without depending on historical workspace paths."""
from pathlib import Path


def gene_vocabulary_path(benchmark_dir: Path) -> Path:
    benchmark_dir = Path(benchmark_dir).resolve()
    local = benchmark_dir.parents[1] / "assets" / "human_v3_genes.parquet"
    if not local.is_file():
        raise FileNotFoundError(
            f"self-contained gene vocabulary is missing: {local}"
        )
    return local

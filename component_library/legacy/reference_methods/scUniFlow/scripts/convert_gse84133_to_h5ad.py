#!/usr/bin/env python3
"""
Convert GSE84133_RAW.tar count tables into h5ad files for scUniFlow.

The archive contains per-sample gzipped CSV files with columns:
cell_id, barcode, assigned_cluster, <gene counts ...>

We intentionally keep this converter independent from the training code.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import tarfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class SampleInfo:
    member_name: str
    sample_id: str
    species: str
    n_cells: int
    genes: tuple[str, ...]


def infer_species(member_name: str) -> str:
    lower = member_name.lower()
    if "human" in lower:
        return "human"
    if "mouse" in lower:
        return "mouse"
    raise ValueError(f"Unable to infer species from filename: {member_name}")


def sample_id_from_name(member_name: str) -> str:
    stem = Path(member_name).name
    suffix = "_umifm_counts.csv.gz"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return stem


def inspect_archive(tar_path: Path) -> list[SampleInfo]:
    infos: list[SampleInfo] = []
    with tarfile.open(tar_path, "r") as tar:
        for member in sorted(tar.getmembers(), key=lambda x: x.name):
            if not member.isfile() or not member.name.endswith(".csv.gz"):
                continue
            species = infer_species(member.name)
            sample_id = sample_id_from_name(member.name)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {member.name}")
            with gzip.open(extracted, "rt", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                genes = tuple(str(x) for x in header[3:])
                n_cells = sum(1 for _ in reader)
            infos.append(
                SampleInfo(
                    member_name=member.name,
                    sample_id=sample_id,
                    species=species,
                    n_cells=n_cells,
                    genes=genes,
                )
            )
    if not infos:
        raise FileNotFoundError(f"No .csv.gz members found in {tar_path}")
    return infos


def select_infos(infos: list[SampleInfo], species: str) -> list[SampleInfo]:
    selected = [info for info in infos if info.species == species]
    if not selected:
        raise ValueError(f"No samples found for species={species!r}")
    reference = selected[0].genes
    for info in selected[1:]:
        if info.genes != reference:
            raise ValueError(
                f"Gene panel mismatch inside species={species}: "
                f"{selected[0].member_name} vs {info.member_name}"
            )
    return selected


def build_species_h5ad(
    tar_path: Path,
    infos: list[SampleInfo],
    species: str,
    output_path: Path,
    chunk_size: int,
) -> None:
    selected = select_infos(infos, species)
    gene_names = list(selected[0].genes)
    obs_parts: list[pd.DataFrame] = []
    matrix_parts: list[sp.csr_matrix] = []

    print(
        f"[convert] species={species} samples={len(selected)} "
        f"cells={sum(x.n_cells for x in selected):,} genes={len(gene_names):,}"
    )

    with tarfile.open(tar_path, "r") as tar:
        for info in selected:
            member = tar.getmember(info.member_name)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {info.member_name}")

            print(f"[read] {info.member_name} -> sample_id={info.sample_id}")
            with gzip.open(extracted, "rt", newline="") as fh:
                chunk_iter = pd.read_csv(fh, chunksize=chunk_size)
                for chunk_idx, chunk in enumerate(chunk_iter, start=1):
                    meta = chunk.iloc[:, :3].copy()
                    meta.columns = ["cell_id", "barcode", "assigned_cluster"]
                    meta["sample_id"] = info.sample_id
                    meta["donor_id"] = info.sample_id
                    meta["species"] = species
                    meta["source_file"] = info.member_name
                    meta["cell_id"] = meta["cell_id"].astype(str)
                    meta["barcode"] = meta["barcode"].astype(str)
                    meta["assigned_cluster"] = meta["assigned_cluster"].astype(str)
                    meta.index = meta["cell_id"]
                    obs_parts.append(meta)

                    counts = chunk.iloc[:, 3:].to_numpy(dtype=np.float32, copy=False)
                    matrix_parts.append(sp.csr_matrix(counts))

                    if chunk_idx % 10 == 0:
                        print(
                            f"  [chunk] {info.sample_id} chunk={chunk_idx} "
                            f"rows={chunk.shape[0]:,}"
                        )

    obs = pd.concat(obs_parts, axis=0)
    if obs.index.duplicated().any():
        dup_count = int(obs.index.duplicated().sum())
        raise ValueError(f"Found duplicated cell_id values: {dup_count}")

    X = sp.vstack(matrix_parts, format="csr", dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_id"))
    var["feature_name"] = var.index.astype(str)
    var["gene_name"] = var.index.astype(str)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["dataset_name"] = "GSE84133"
    adata.uns["species"] = species
    adata.uns["source_tar"] = str(tar_path)
    adata.uns["included_files"] = [info.member_name for info in selected]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    print(f"[saved] {output_path} shape={adata.n_obs:,} x {adata.n_vars:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GSE84133_RAW.tar into h5ad files for scUniFlow."
    )
    parser.add_argument(
        "--tar_path",
        type=Path,
        default=Path("data/GSE84133_RAW.tar"),
        help="Path to GSE84133_RAW.tar",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/GSE84133_h5ad"),
        help="Directory for converted h5ad files",
    )
    parser.add_argument(
        "--species",
        choices=["human", "mouse", "both"],
        default="both",
        help="Which species subset(s) to export",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Rows per chunk when reading CSV data",
    )
    args = parser.parse_args()

    infos = inspect_archive(args.tar_path)
    print("[inspect] archive summary")
    for info in infos:
        print(
            f"  - {info.member_name}: species={info.species} "
            f"sample_id={info.sample_id} cells={info.n_cells:,} genes={len(info.genes):,}"
        )

    targets = ["human", "mouse"] if args.species == "both" else [args.species]
    for species in targets:
        output_path = args.out_dir / f"gse84133_{species}.h5ad"
        build_species_h5ad(
            tar_path=args.tar_path,
            infos=infos,
            species=species,
            output_path=output_path,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()

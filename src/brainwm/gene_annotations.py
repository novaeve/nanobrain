"""Versioned gene annotation and optional BrainBeacon token features.

The current healthy corpus uses a human Ensembl vocabulary.  This module keeps
the annotation table separate from that vocabulary so adding mouse/monkey
records does not silently change existing gene indices or checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

MISSING = "<missing>"
UNKNOWN = "<unk>"
EXTERNAL_EMBEDDING_FORMAT = "brainwm_external_gene_embedding_artifact_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_gene_embeddings(
    path: str | Path,
    expected_n_genes: int,
) -> tuple[np.ndarray, dict[str, object] | None]:
    """Load a legacy array or a provenance-checked embedding artifact.

    Artifact directories are the preferred interface. They bind the array to
    a specific BrainWM vocabulary, source checkpoint, exact Ensembl mapping,
    and explicit zero-vector policy for genes absent from the source model.
    Plain ``.npy`` files remain supported for checkpoint compatibility but do
    not carry auditable provenance.
    """
    source = Path(path)
    manifest: dict[str, object] | None = None
    source_token_ids: np.ndarray | None = None
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("external gene embedding artifact lacks manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != EXTERNAL_EMBEDDING_FORMAT:
            raise ValueError("unexpected external gene embedding artifact format")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("external gene embedding manifest lacks file hashes")
        for filename, expected_hash in files.items():
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("external gene embedding artifact has an unsafe filename")
            file_path = source / filename
            if not file_path.is_file():
                raise ValueError(f"external gene embedding artifact lacks {filename}")
            if not isinstance(expected_hash, str) or _sha256(file_path) != expected_hash:
                raise ValueError(
                    f"external gene embedding artifact hash mismatch: {filename}"
                )
        array_meta = manifest.get("array")
        mapping_meta = manifest.get("mapping")
        if not isinstance(array_meta, dict) or not isinstance(mapping_meta, dict):
            raise ValueError("external gene embedding manifest lacks array/mapping metadata")
        array_filename = array_meta.get("filename")
        token_filename = mapping_meta.get("source_token_ids_filename")
        if not isinstance(array_filename, str) or Path(array_filename).name != array_filename:
            raise ValueError("external gene embedding filename is invalid")
        if not isinstance(token_filename, str) or Path(token_filename).name != token_filename:
            raise ValueError("external source-token filename is invalid")
        if array_filename not in files or token_filename not in files:
            raise ValueError("external embedding arrays must be covered by file hashes")
        external = np.load(source / array_filename, allow_pickle=False)
        source_token_ids = np.load(source / token_filename, allow_pickle=False)
        if mapping_meta.get("unknown_policy") != "zero":
            raise ValueError("external gene embedding artifact must declare zero unknowns")
        if (
            source_token_ids.ndim != 1
            or len(source_token_ids) != expected_n_genes
            or not np.issubdtype(source_token_ids.dtype, np.integer)
        ):
            raise ValueError("external source-token IDs must have shape [n_genes]")
        matched = source_token_ids >= 0
        if int(matched.sum()) != int(mapping_meta.get("matched_genes", -1)):
            raise ValueError("external embedding coverage disagrees with source-token IDs")
        if np.any(external[~matched] != 0):
            raise ValueError("unmatched external gene embedding rows must be exactly zero")
    else:
        external = np.load(source, allow_pickle=False)

    if external.ndim != 2 or external.shape[0] != expected_n_genes:
        raise ValueError("external gene embedding must have shape [n_genes, dim]")
    if external.shape[1] < 1:
        raise ValueError("external gene embedding dimension must be positive")
    if not np.issubdtype(external.dtype, np.floating):
        raise ValueError("external gene embedding must use a floating dtype")
    if not np.isfinite(external).all():
        raise ValueError("external gene embedding contains non-finite values")
    if manifest is not None:
        array_meta = manifest["array"]
        declared_shape = tuple(int(value) for value in array_meta.get("shape", []))
        if declared_shape != external.shape:
            raise ValueError("external gene embedding shape disagrees with manifest")
        if int(manifest.get("n_genes", -1)) != expected_n_genes:
            raise ValueError("external embedding vocabulary size disagrees with model")
    return np.asarray(external), manifest


def normalize_ensembl(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().split(".", 1)[0]
    return text or None


def read_gencode_table(path: str | Path, species: str) -> pd.DataFrame:
    """Read one of the supplied GENCODE TSVs into a stable common schema."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    required = {"gene_name", "ensembl_id", "gene_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if species not in {"human", "mouse", "macaque", "marmoset"}:
        raise ValueError(f"unsupported species: {species}")
    result = pd.DataFrame({
        "species": species,
        "ensembl_id": frame["ensembl_id"].map(normalize_ensembl),
        "gene_symbol": frame["gene_name"].astype("string").str.strip(),
        "gene_type": frame["gene_type"].astype("string").str.strip(),
        "chromosome": (
            frame["chromosome"].astype("string").str.strip()
            if "chromosome" in frame
            else frame["chromosome_location"].astype("string").str.split(":").str[0]
        ),
        "annotation_source": str(path.resolve()),
    })
    result = result.dropna(subset=["ensembl_id"])
    if result["ensembl_id"].duplicated().any():
        duplicate = result.loc[result["ensembl_id"].duplicated(), "ensembl_id"].iloc[0]
        raise ValueError(f"{path} contains duplicate Ensembl ID: {duplicate}")
    return result.reset_index(drop=True)


def _category_ids(values: pd.Series) -> tuple[dict[str, int], np.ndarray]:
    vocab = {MISSING: 0, UNKNOWN: 1}
    for value in sorted(values.dropna().astype(str).unique()):
        if value not in vocab:
            vocab[value] = len(vocab)
    ids = np.asarray([
        vocab.get(str(value), vocab[UNKNOWN]) if pd.notna(value) else vocab[MISSING]
        for value in values
    ], dtype=np.int64)
    return vocab, ids


def build_annotation_frame(
    genes_path: str | Path,
    annotation_tables: Iterable[tuple[str, str | Path]],
    orthology_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join annotations to the current vocabulary without changing indices.

    ``gene_index`` is populated for genes already present in ``genes.parquet``;
    non-human rows remain available for a future unified cross-species vocab.
    An optional orthology table may provide ``ensembl_id`` and
    ``homology_group`` columns.  No orthology is inferred when it is absent.
    """
    genes = pd.read_parquet(genes_path, columns=[
        "gene_index", "ensembl_id", "gene_symbol",
    ]).copy()
    genes["ensembl_id"] = genes["ensembl_id"].map(normalize_ensembl)
    genes["species"] = np.where(
        genes["ensembl_id"].astype("string").str.startswith("ENSMUSG"),
        "mouse", "human",
    )
    tables = [read_gencode_table(path, species) for species, path in annotation_tables]
    if not tables:
        raise ValueError("at least one annotation table is required")
    annotations = pd.concat(tables, ignore_index=True)
    if orthology_path:
        orthology = pd.read_csv(orthology_path, sep="\t", dtype="string")
        required = {"ensembl_id", "homology_group"}
        missing = required - set(orthology.columns)
        if missing:
            raise ValueError(f"orthology table is missing columns: {sorted(missing)}")
        orthology = orthology[["ensembl_id", "homology_group"]].copy()
        orthology["ensembl_id"] = orthology["ensembl_id"].map(normalize_ensembl)
        if orthology["ensembl_id"].isna().any():
            raise ValueError("orthology table contains an empty Ensembl ID")
        orthology["homology_group"] = orthology["homology_group"].astype("string").str.strip()
        if orthology["homology_group"].isna().any() or orthology["homology_group"].eq("").any():
            raise ValueError("orthology table contains an empty homology group")
        if orthology["ensembl_id"].duplicated().any():
            raise ValueError("orthology table contains duplicate Ensembl IDs")
        annotations = annotations.merge(orthology, on="ensembl_id", how="left")
        orthology_rows = len(orthology)
    else:
        annotations["homology_group"] = pd.NA
        orthology_rows = 0

    gene_lookup = genes.rename(columns={
        "gene_symbol": "vocabulary_gene_symbol",
        "species": "vocabulary_species",
    })
    result = annotations.merge(gene_lookup, on=["ensembl_id"], how="left")
    result["gene_index"] = result["gene_index"].astype("Int64")
    result = result[
        ["gene_index", "species", "ensembl_id", "gene_symbol", "gene_type",
         "chromosome", "homology_group", "annotation_source",
         "vocabulary_gene_symbol", "vocabulary_species"]
    ].sort_values(["species", "ensembl_id"], kind="stable").reset_index(drop=True)

    # Arrays are intentionally only for the existing global vocabulary.  They
    # make enabling type/homology features cheap while retaining all species
    # rows in the tabular artifact for the later unified vocabulary build.
    indexed = result[result["gene_index"].notna()].copy()
    max_index = int(genes["gene_index"].max())
    type_vocab, _ = _category_ids(result["gene_type"])
    homology_values = result["homology_group"].fillna(MISSING)
    homology_vocab, _ = _category_ids(homology_values)
    type_ids = np.zeros(max_index + 1, dtype=np.int64)
    homology_ids = np.zeros(max_index + 1, dtype=np.int64)
    for row in indexed.itertuples(index=False):
        index = int(row.gene_index)
        type_ids[index] = type_vocab.get(str(row.gene_type), type_vocab[UNKNOWN])
        group = MISSING if pd.isna(row.homology_group) else str(row.homology_group)
        homology_ids[index] = homology_vocab.get(group, homology_vocab[UNKNOWN])

    stats = {
        "genes_path": str(Path(genes_path).resolve()),
        "rows": len(result),
        "species": {str(k): int(v) for k, v in result["species"].value_counts().items()},
        "vocabulary_genes": len(indexed),
        "vocabulary_coverage": float(len(indexed) / max(len(genes), 1)),
        "gene_type_vocab_size": len(type_vocab),
        "homology_group_vocab_size": len(homology_vocab),
        "orthology_provided": bool(orthology_path),
        "orthology_input_rows": orthology_rows,
        "annotation_homology_genes": int(result["homology_group"].notna().sum()),
        "annotation_homology_coverage": float(
            result["homology_group"].notna().mean()
        ),
        "species_homology_coverage": {
            str(species): {
                "genes": len(frame),
                "homology_genes": int(frame["homology_group"].notna().sum()),
                "coverage": float(frame["homology_group"].notna().mean()),
            }
            for species, frame in result.groupby("species", sort=True)
        },
        "vocabulary_homology_genes": int(indexed["homology_group"].notna().sum()),
        "vocabulary_homology_coverage": float(
            indexed["homology_group"].notna().sum() / max(len(genes), 1)
        ),
    }
    result.attrs["type_vocab"] = type_vocab
    result.attrs["homology_vocab"] = homology_vocab
    result.attrs["type_ids"] = type_ids
    result.attrs["homology_ids"] = homology_ids
    result.attrs["stats"] = stats
    return result, stats


class GeneTokenFeatureEncoder(nn.Module):
    """Optional additive gene features for cross-species tokenization.

    The module is only constructed when explicitly enabled.  This preserves
    strict loading of all existing Healthy checkpoints and makes missing
    BioMart/ESM2 assets an explicit configuration error rather than an implicit
    fallback to fabricated embeddings.
    """

    def __init__(self, feature_dir: str | Path, gene_dim: int,
                 external_embedding_path: str = "", *,
                 include_gene_type: bool = True,
                 include_homology_group: bool = False,
                 expected_n_genes: int | None = None):
        super().__init__()
        root = Path(feature_dir)
        meta = json.loads((root / "feature_vocab.json").read_text())

        if expected_n_genes is not None:
            if isinstance(expected_n_genes, bool) or expected_n_genes < 1:
                raise ValueError("expected_n_genes must be a positive integer")
            expected_n_genes = int(expected_n_genes)

        def category_size(size_key: str, vocab_key: str) -> int:
            # Older annotation artifacts stored only ``*_vocab_size``;
            # unified human/mouse artifacts store the complete vocab map.
            value = meta.get(size_key)
            if value is None:
                vocab = meta.get(vocab_key)
                if not isinstance(vocab, dict):
                    raise ValueError(
                        f"feature vocabulary must contain {size_key!r} or "
                        f"a mapping at {vocab_key!r}"
                    )
                value = len(vocab)
            value = int(value)
            if value < 1:
                raise ValueError(f"{size_key} must be positive")
            return value

        type_size = category_size("gene_type_vocab_size", "gene_type_vocab")
        homology_size = category_size(
            "homology_group_vocab_size", "homology_group_vocab"
        )
        self.gene_type = (
            nn.Embedding(type_size, gene_dim) if include_gene_type else None
        )
        self.homology_group = (
            nn.Embedding(homology_size, gene_dim)
            if include_homology_group else None
        )
        arrays = np.load(root / "feature_arrays.npz")
        if "gene_type_ids" not in arrays or "homology_group_ids" not in arrays:
            raise ValueError(
                "feature_arrays.npz must contain gene_type_ids and homology_group_ids"
            )
        type_ids = np.asarray(arrays["gene_type_ids"], dtype=np.int64)
        homology_ids = np.asarray(arrays["homology_group_ids"], dtype=np.int64)
        if type_ids.ndim != 1 or homology_ids.ndim != 1:
            raise ValueError("gene feature ID arrays must be one-dimensional")
        if type_ids.shape != homology_ids.shape:
            raise ValueError("gene feature ID arrays must have the same length")
        array_n_genes = int(type_ids.shape[0])
        declared_n_genes = meta.get("n_genes")
        if declared_n_genes is not None:
            if isinstance(declared_n_genes, bool):
                raise ValueError("feature vocabulary n_genes must be an integer")
            try:
                declared_n_genes = int(declared_n_genes)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "feature vocabulary n_genes must be an integer"
                ) from exc
            if declared_n_genes < 1:
                raise ValueError("feature vocabulary n_genes must be positive")
            if declared_n_genes != array_n_genes:
                raise ValueError(
                    "feature vocabulary n_genes does not match feature array length"
                )
        if expected_n_genes is not None and array_n_genes != expected_n_genes:
            raise ValueError(
                "gene feature array length does not match model n_genes: "
                f"expected {expected_n_genes}, got {array_n_genes}"
            )
        if np.any(type_ids < 0) or np.any(type_ids >= type_size):
            raise ValueError("gene_type_ids contain an out-of-range category")
        if np.any(homology_ids < 0) or np.any(homology_ids >= homology_size):
            raise ValueError("homology_group_ids contain an out-of-range category")
        self.register_buffer("gene_type_ids", torch.as_tensor(arrays["gene_type_ids"], dtype=torch.long))
        self.register_buffer("homology_group_ids", torch.as_tensor(arrays["homology_group_ids"], dtype=torch.long))
        self.external = None
        self.external_embedding_manifest: dict[str, object] | None = None
        if external_embedding_path:
            external, manifest = load_external_gene_embeddings(
                external_embedding_path, len(self.gene_type_ids)
            )
            # Adding this single factor must not perturb initialization of the
            # value encoder and downstream backbone modules created next.
            with torch.random.fork_rng(devices=[]):
                self.external = nn.Linear(
                    int(external.shape[1]), gene_dim, bias=False
                )
            self.register_buffer("external_embeddings", torch.as_tensor(external, dtype=torch.float32))
            self.external_embedding_manifest = manifest
        if self.gene_type is None and self.homology_group is None and self.external is None:
            raise ValueError("at least one gene token feature must be enabled")

    def forward(self, gene_index: Tensor) -> Tensor:
        if torch.any(gene_index < 0) or torch.any(gene_index >= self.gene_type_ids.numel()):
            raise ValueError("gene index is outside gene-token feature arrays")
        value = torch.zeros(
            (*gene_index.shape, self.gene_type.embedding_dim
             if self.gene_type is not None
             else self.homology_group.embedding_dim
             if self.homology_group is not None
             else self.external.out_features),
            device=gene_index.device,
            dtype=(
                self.gene_type.weight.dtype if self.gene_type is not None
                else self.homology_group.weight.dtype if self.homology_group is not None
                else self.external.weight.dtype
            ),
        )
        if self.gene_type is not None:
            value = value + self.gene_type(self.gene_type_ids[gene_index])
        if self.homology_group is not None:
            value = value + self.homology_group(
                self.homology_group_ids[gene_index]
            )
        if self.external is not None:
            value = value + self.external(self.external_embeddings[gene_index])
        return value

"""Bridge from the Healthy indexed H5AD reader to strict model tensors."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .types import SparseExpressionBatch, WorldModelBatch


MISSING = "<missing>"
UNKNOWN = "<unk>"


# The data artifacts use explicit species IDs.  Keeping the small default map
# here gives standalone metadata normalization a deterministic policy while
# callers with a versioned vocabulary can pass their own map.
DEFAULT_SPECIES_IDS: dict[str, int] = {
    "human": 0,
    "mouse": 1,
    "macaque": 2,
    "marmoset": 3,
}

_SPECIES_ALIASES = {
    "homo sapiens": "human",
    "h. sapiens": "human",
    "hsapiens": "human",
    "h_sapiens": "human",
    "human": "human",
    "mus musculus": "mouse",
    "m. musculus": "mouse",
    "mmusculus": "mouse",
    "m_musculus": "mouse",
    "mouse": "mouse",
    "macaca mulatta": "macaque",
    "m. mulatta": "macaque",
    "macaque": "macaque",
    "callithrix jacchus": "marmoset",
    "c. jacchus": "marmoset",
    "marmoset": "marmoset",
}

_ASSAY_ALIASES = {
    "scrna": "scRNA",
    "singlecell": "scRNA",
    "single cell": "scRNA",
    "single-cell": "scRNA",
    "sc": "scRNA",
    "snrna": "snRNA",
    "singlenucleus": "snRNA",
    "single nucleus": "snRNA",
    "single-nucleus": "snRNA",
    "sn": "snRNA",
    "spatialrna": "spatialRNA",
    "spatial rna": "spatialRNA",
    "spatial-rna": "spatialRNA",
    "spatial": "spatialRNA",
}

_TIME_POINT_COLUMNS = (
    "time_point_days",
    "timepoint_days",
    "time_days",
    "timepoint",
    "time",
    "age_days",
    "age",
    "timepoint_raw",
    "age_raw",
    "developmental_stage_raw",
    "developmental_stage",
    "stage",
)
_POINT_ONLY_COLUMNS = {
    "time_point_days",
    "timepoint_days",
    "time_days",
    "age_days",
}
_TIME_LOWER_COLUMNS = (
    "time_lower_days",
    "timepoint_lower_days",
    "age_lower_days",
)
_TIME_UPPER_COLUMNS = (
    "time_upper_days",
    "timepoint_upper_days",
    "age_upper_days",
)

# Anatomical annotations occur under several source-specific names.  Keep the
# ontology identifier separate from free-text region/layer labels: only the
# identifier is eligible for the model's existing UBERON categorical
# embedding, while the text fields remain available for audits and downstream
# tasks.  These aliases are deliberately conservative; no free-text region is
# guessed to be a UBERON ID.
_UBERON_ID_COLUMNS = (
    "uberon_id",
    "uberon",
    "uberon_identifier",
    "uberon_term_id",
    "anatomical_ontology_id",
    "anatomical_region_id",
)
_UBERON_TERM_COLUMNS = (
    "uberon_term",
    "anatomical_region_term",
    "anatomy_term",
)
_ANATOMICAL_REGION_COLUMNS = (
    "anatomical_region",
    "original_anatomical_region",
    "brain_region",
    "brain_area",
    "region",
    "location",
)
_ANATOMICAL_LAYER_COLUMNS = (
    "anatomical_layer",
    "layer",
    "cortical_layer",
)


def _missing_scalar(value: Any) -> bool:
    """Return a scalar missingness result without array truth-value errors."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if np.isscalar(result) else False


def _normalize_text(value: Any) -> str | None:
    if _missing_scalar(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_uberon_id(value: Any) -> str | None:
    """Normalize only the spelling of an explicit UBERON identifier."""
    text = _normalize_text(value)
    if text is None:
        return None
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        if prefix.strip().lower() == "uberon":
            return f"UBERON:{suffix.strip()}"
    return text


def _canonicalize_anatomical_columns(frame: pd.DataFrame) -> None:
    """Coalesce source-specific anatomical aliases on a private frame copy.

    A missing annotation remains missing.  In particular, a term such as
    ``cortex`` is never converted into a fabricated UBERON identifier.
    Contradictory explicit identifiers fail closed through the same policy as
    assay/species aliases.
    """
    if any(column in frame for column in _UBERON_ID_COLUMNS):
        frame["uberon_id"] = _coalesced_text_column(
            frame, _UBERON_ID_COLUMNS, _normalize_uberon_id, "uberon_id"
        )
    if any(column in frame for column in _UBERON_TERM_COLUMNS):
        frame["uberon_term"] = _coalesced_text_column(
            frame, _UBERON_TERM_COLUMNS, _normalize_text, "uberon_term"
        )
    if any(column in frame for column in _ANATOMICAL_REGION_COLUMNS):
        frame["anatomical_region"] = _coalesced_text_column(
            frame,
            _ANATOMICAL_REGION_COLUMNS,
            _normalize_text,
            "anatomical_region",
        )
    if any(column in frame for column in _ANATOMICAL_LAYER_COLUMNS):
        frame["anatomical_layer"] = _coalesced_text_column(
            frame,
            _ANATOMICAL_LAYER_COLUMNS,
            _normalize_text,
            "anatomical_layer",
        )


def _normalize_species(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    key = text.lower().replace("_", " ").strip()
    return _SPECIES_ALIASES.get(key, text)


def _normalize_assay(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    key = text.lower().replace("_", " ").strip()
    compact = key.replace(" ", "")
    return _ASSAY_ALIASES.get(key, _ASSAY_ALIASES.get(compact, text))


def _coalesced_text_column(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    normalizer,
    canonical: str,
) -> pd.Series:
    """Coalesce aliases and reject contradictory non-null values."""
    present = [column for column in columns if column in frame]
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in present:
        values = frame[column].map(normalizer).astype("string")
        conflict = (
            result.notna()
            & values.notna()
            & result.ne(values).fillna(False)
        )
        if bool(conflict.any()):
            rows = frame.index[conflict].tolist()[:5]
            raise ValueError(
                f"conflicting {canonical} aliases in rows {rows}: "
                f"{present}"
            )
        result = result.fillna(values)
    return result


def _species_id_map(
    species_vocab: Mapping[str, int] | None,
) -> tuple[dict[str, int], dict[int, str]]:
    raw = DEFAULT_SPECIES_IDS if species_vocab is None else species_vocab
    forward: dict[str, int] = {}
    reverse: dict[int, str] = {}
    for name, value in raw.items():
        species = _normalize_species(name)
        if species is None:
            raise ValueError("species vocabulary contains an empty name")
        try:
            numeric_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"species ID is not an integer: {name}={value!r}") from exc
        if not math.isfinite(numeric_float) or numeric_float != math.floor(numeric_float):
            raise ValueError(f"species ID must be a nonnegative integer: {name}={value!r}")
        numeric = int(numeric_float)
        if numeric < 0:
            raise ValueError(f"species ID must be a nonnegative integer: {name}={value!r}")
        if species in forward and forward[species] != numeric:
            raise ValueError(f"species vocabulary aliases disagree for {species}")
        if numeric in reverse and reverse[numeric] != species:
            raise ValueError(f"species vocabulary reuses ID {numeric}")
        forward[species] = numeric
        reverse[numeric] = species
    return forward, reverse


def _time_result(
    point: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    basis: str | None = None,
    status: str = "missing",
    life_stage: str = "unknown",
) -> dict[str, Any]:
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "basis": basis,
        "status": status,
        "life_stage": life_stage,
    }


def _stage_for_days(point: float) -> str:
    if point < 0:
        return "prenatal"
    if point >= 18.0 * 365.2425:
        return "adult"
    return "postnatal"


def _parse_time_value(value: Any, field: str) -> dict[str, Any]:
    """Parse one time/age alias into the contract's day-based fields.

    Numeric values are already interpreted as days (the model contract's
    canonical unit). Strings with units use the shared conservative parser;
    categorical developmental stages remain explicitly non-numeric.
    """
    if _missing_scalar(value):
        return _time_result()
    if isinstance(value, (bool, np.bool_)):
        return _time_result(status="unparsed")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return _time_result(
                numeric,
                numeric,
                numeric,
                "numeric_days",
                "exact",
                _stage_for_days(numeric),
            )
        return _time_result(status="unparsed")

    text = str(value).strip()
    if not text:
        return _time_result()

    # Reuse the data-layer parser when it is available, but keep this module
    # importable as a standalone brainwm package.
    try:
        from brainwm_data.age import parse_age
    except ModuleNotFoundError:
        parse_age = None
    if parse_age is not None:
        parsed = parse_age(text)
        if (
            parsed.get("time_point_days") is not None
            or parsed.get("time_lower_days") is not None
            or parsed.get("time_upper_days") is not None
            or parsed.get("life_stage") != "unknown"
        ):
            return _time_result(
                parsed.get("time_point_days"),
                parsed.get("time_lower_days"),
                parsed.get("time_upper_days"),
                parsed.get("time_basis"),
                parsed.get("time_status", "unparsed"),
                parsed.get("life_stage", "unknown"),
            )

    lower = text.lower()
    if lower in {"adult", "adult stage", "mature"}:
        return _time_result(status="categorical_only", life_stage="adult")
    if lower in {"prenatal", "embryonic", "fetal"}:
        return _time_result(status="categorical_only", life_stage="prenatal")
    if lower in {"postnatal", "juvenile", "infant", "newborn"}:
        return _time_result(status="categorical_only", life_stage="postnatal")

    # Common mouse developmental labels are retained as exact day offsets
    # relative to birth; this is only applied to explicit P/E prefixes.
    match = re.fullmatch(r"([pe])\s*([0-9]+(?:\.[0-9]+)?)", lower)
    if match:
        value_days = float(match.group(2))
        point = value_days if match.group(1) == "p" else value_days - 266.0
        basis = "postnatal_day" if match.group(1) == "p" else "embryonic_day"
        return _time_result(
            point, point, point, basis, "exact", _stage_for_days(point)
        )

    # A bare numeric string is common in exported timepoint columns.  Treat it
    # like a numeric value, while leaving arbitrary labels untouched.
    if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", text):
        numeric = float(text)
        return _time_result(
            numeric,
            numeric,
            numeric,
            "numeric_days",
            "exact",
            _stage_for_days(numeric),
        )
    return _time_result(status="unparsed")


def _merge_time_result(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    row: Any,
    field: str,
) -> None:
    """Merge parsed aliases while rejecting contradictory numeric evidence."""
    for key in ("point", "lower", "upper"):
        left, right = current[key], candidate[key]
        if left is not None and right is not None and not math.isclose(
            float(left), float(right), rel_tol=1.0e-5, abs_tol=1.0e-4
        ):
            raise ValueError(
                f"conflicting time aliases at row {row}, field {field}: "
                f"{left!r} != {right!r}"
            )
        if left is None and right is not None:
            current[key] = float(right)
    if current["basis"] is None and candidate["basis"] is not None:
        current["basis"] = candidate["basis"]
    if current["status"] in {"missing", "unparsed"} and candidate["status"] not in {
        "missing",
        "unparsed",
    }:
        current["status"] = candidate["status"]
    if current["life_stage"] == "unknown" and candidate["life_stage"] != "unknown":
        current["life_stage"] = candidate["life_stage"]


def _canonicalize_time_columns(frame: pd.DataFrame) -> None:
    """Add/normalize canonical time columns in-place on a private frame copy."""
    point_sources = [column for column in _TIME_POINT_COLUMNS if column in frame]
    lower_sources = [column for column in _TIME_LOWER_COLUMNS if column in frame]
    upper_sources = [column for column in _TIME_UPPER_COLUMNS if column in frame]
    results = [_time_result() for _ in range(len(frame))]

    for column in point_sources:
        for position, (index, value) in enumerate(frame[column].items()):
            candidate = _parse_time_value(value, column)
            if candidate["status"] == "missing":
                continue
            # Explicit ``*_days`` point columns carry only a point estimate.
            # Treating that point as an exact lower/upper bound incorrectly
            # conflicts with valid range metadata such as ``65-77y``.
            if column in _POINT_ONLY_COLUMNS:
                candidate["lower"] = None
                candidate["upper"] = None
            _merge_time_result(results[position], candidate, row=index, field=column)

    # Explicit lower/upper aliases are merged independently; they must not
    # accidentally become a point estimate when only a bound is available.
    for column, key in [
        *[(column, "lower") for column in lower_sources],
        *[(column, "upper") for column in upper_sources],
    ]:
        for position, (index, value) in enumerate(frame[column].items()):
            candidate = _parse_time_value(value, column)
            bound = candidate["point"]
            if bound is None:
                bound = candidate[key]
            if bound is None:
                continue
            current = results[position][key]
            if current is not None and not math.isclose(
                float(current), float(bound), rel_tol=1.0e-5, abs_tol=1.0e-4
            ):
                raise ValueError(
                    f"conflicting time aliases at row {index}, field {column}: "
                    f"{current!r} != {bound!r}"
                )
            results[position][key] = float(bound)

    existing_status = (
        frame["time_status"] if "time_status" in frame else pd.Series(
            pd.NA, index=frame.index, dtype="string"
        )
    )
    existing_basis = (
        frame["time_basis"] if "time_basis" in frame else pd.Series(
            pd.NA, index=frame.index, dtype="string"
        )
    )
    existing_stage = (
        frame["life_stage"] if "life_stage" in frame else pd.Series(
            pd.NA, index=frame.index, dtype="string"
        )
    )
    points: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    statuses: list[str] = []
    bases: list[str | None] = []
    stages: list[str] = []
    for position, index in enumerate(frame.index):
        result = results[position]
        point = result["point"]
        lower = result["lower"]
        upper = result["upper"]
        if point is not None and lower is not None and point < lower - 1.0e-4:
            raise ValueError(f"time point precedes lower bound at row {index}")
        if point is not None and upper is not None and point > upper + 1.0e-4:
            raise ValueError(f"time point exceeds upper bound at row {index}")
        points.append(float(point) if point is not None else np.nan)
        lowers.append(float(lower) if lower is not None else np.nan)
        uppers.append(float(upper) if upper is not None else np.nan)
        status = _normalize_text(existing_status.iloc[position])
        statuses.append(status or result["status"] or "missing")
        bases.append(_normalize_text(existing_basis.iloc[position]) or result["basis"])
        stages.append(
            _normalize_text(existing_stage.iloc[position])
            or result["life_stage"]
            or "unknown"
        )
    frame["time_point_days"] = pd.Series(points, index=frame.index, dtype="float64")
    frame["time_lower_days"] = pd.Series(lowers, index=frame.index, dtype="float64")
    frame["time_upper_days"] = pd.Series(uppers, index=frame.index, dtype="float64")
    frame["time_status"] = pd.Series(statuses, index=frame.index, dtype="string")
    frame["time_basis"] = pd.Series(bases, index=frame.index, dtype="string")
    frame["life_stage"] = pd.Series(stages, index=frame.index, dtype="string")

    # Keep raw source values available under stable names for provenance.
    for canonical, aliases in (
        ("age_raw", ("age_raw", "age")),
        ("timepoint_raw", ("timepoint_raw", "timepoint", "time")),
        (
            "developmental_stage_raw",
            ("developmental_stage_raw", "developmental_stage", "stage"),
        ),
    ):
        if canonical not in frame:
            values = pd.Series(pd.NA, index=frame.index, dtype="string")
            for alias in aliases:
                if alias in frame:
                    values = values.fillna(frame[alias].map(_normalize_text).astype("string"))
            frame[canonical] = values


def canonicalize_metadata(
    metadata: pd.DataFrame,
    *,
    species_vocab: Mapping[str, int] | None = None,
    require_species: bool = True,
) -> pd.DataFrame:
    """Return a canonical, non-mutating metadata frame for model tensorization.

    The helper accepts the human index schema and the mouse molecular/spatial
    schemas. It preserves source columns while adding canonical ``assay_type``,
    explicit ``species``/``species_id``, day-based time fields, anatomical
    aliases (``uberon_id``, ``uberon_term``, ``anatomical_region`` and
    ``anatomical_layer``), and missing spatial coordinate columns.
    Contradictory aliases fail closed; missing time or anatomy remains missing
    rather than being inferred from dataset names.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    frame = metadata.copy()

    # Preserve the historical API contract for callers that did not provide
    # any species signal at all.  Alias-only frames (``organism``/
    # ``species_name``) and explicit ``species_id`` remain valid inputs to the
    # canonicalizer; a completely absent signal is a schema error rather than
    # an unknown species value.
    if require_species and not any(
        column in frame for column in ("species", "organism", "species_name", "species_id")
    ):
        raise KeyError("metadata must contain an explicit species column")

    frame["assay_type"] = _coalesced_text_column(
        frame, ("assay_type", "assay"), _normalize_assay, "assay_type"
    )

    species_to_id, id_to_species = _species_id_map(species_vocab)
    species_values = _coalesced_text_column(
        frame,
        ("species", "organism", "species_name"),
        _normalize_species,
        "species",
    )
    if "species_id" in frame:
        raw_ids = pd.to_numeric(frame["species_id"], errors="coerce")
        non_missing = frame["species_id"].map(lambda value: not _missing_scalar(value))
        invalid = non_missing & (
            raw_ids.isna() | (raw_ids % 1 != 0) | (raw_ids < 0)
        )
        if bool(invalid.any()):
            raise ValueError("species_id must contain nonnegative integer values")
        species_ids = raw_ids.astype("Int64")
    else:
        species_ids = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    for position, index in enumerate(frame.index):
        species = species_values.iloc[position]
        numeric = species_ids.iloc[position]
        numeric_value = None if pd.isna(numeric) else int(numeric)
        if species is None or pd.isna(species):
            if numeric_value is not None:
                if numeric_value not in id_to_species:
                    raise ValueError(f"species_id {numeric_value} has no registered species")
                species = id_to_species[numeric_value]
                species_values.iloc[position] = species
        else:
            species = str(species)
            if species not in species_to_id:
                if require_species:
                    raise ValueError(f"species is not present in the vocabulary: {species}")
            else:
                expected = species_to_id[species]
                if numeric_value is not None and numeric_value != expected:
                    raise ValueError(
                        f"species/species_id mismatch at row {index}: "
                        f"{species!r} != {numeric_value}"
                    )
                species_ids.iloc[position] = expected
                species_values.iloc[position] = species
        if (species is None or pd.isna(species)) and require_species:
            raise ValueError(f"metadata row {index} lacks an explicit species")

    frame["species"] = species_values.astype("string")
    frame["species_id"] = species_ids.astype("Int64")
    _canonicalize_time_columns(frame)
    _canonicalize_anatomical_columns(frame)
    for column in ("spatial_x", "spatial_y"):
        if column not in frame:
            frame[column] = np.nan
    if "sample_key" not in frame:
        for alias in ("sample_id", "library_label"):
            if alias in frame:
                frame["sample_key"] = frame[alias].map(_normalize_text).astype("string")
                break
    return frame


# ``normalize_metadata`` is a descriptive compatibility alias for callers that
# use the older terminology; both names intentionally share one implementation.
normalize_metadata = canonicalize_metadata


@dataclass(frozen=True)
class TfidfAuxiliarySpec:
    """Source-fitted TF-IDF transform used by the optional model branch.

    The sidecar stores only a fixed global-gene panel and source-fit IDF
    statistics.  Per-cell values are derived from the raw sparse batch at
    read time, so the full expression input remains untouched and no target
    labels (or target distribution statistics) can enter the transform.
    """

    panel_gene_index: np.ndarray
    idf: np.ndarray
    tf_scale: float = 1.0e4
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_dim: int,
        n_genes: int,
        expected_fit_split: str | None = None,
        expected_fit_assay_types: list[str] | None = None,
    ) -> "TfidfAuxiliarySpec":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"optional modality sidecar not found: {path}")
        if path.suffix.lower() != ".json":
            raise ValueError(
                "TF-IDF optional modality sidecar must be a JSON file; "
                f"got {path.suffix or '<no suffix>'}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("TF-IDF sidecar root must be a JSON object")
        if raw.get("format") != "brainwm_tfidf_auxiliary_v1":
            raise ValueError(
                "unsupported optional modality sidecar format: "
                f"{raw.get('format')!r}"
            )
        if raw.get("target_labels_used_for_fit") is not False:
            raise ValueError(
                "TF-IDF sidecar must explicitly declare target_labels_used_for_fit=false"
            )
        if raw.get("target_expression_used_for_fit") is not False:
            raise ValueError(
                "TF-IDF sidecar must explicitly declare target_expression_used_for_fit=false"
            )
        if (
            expected_fit_split is not None
            and raw.get("fit_split") != expected_fit_split
        ):
            raise ValueError(
                "TF-IDF sidecar fit_split mismatch: "
                f"{raw.get('fit_split')!r} != {expected_fit_split!r}"
            )
        if expected_fit_assay_types is not None:
            actual_assays = raw.get("fit_assay_types")
            if actual_assays != expected_fit_assay_types:
                raise ValueError(
                    "TF-IDF sidecar fit_assay_types mismatch: "
                    f"{actual_assays!r} != {expected_fit_assay_types!r}"
                )
        panel = np.asarray(raw.get("panel_gene_index", []), dtype=np.int64)
        idf = np.asarray(raw.get("idf", []), dtype=np.float32)
        if panel.ndim != 1 or idf.ndim != 1:
            raise ValueError("TF-IDF panel_gene_index and idf must be one-dimensional")
        if panel.size != expected_dim or idf.size != expected_dim:
            raise ValueError(
                "TF-IDF sidecar dimension mismatch: "
                f"panel={panel.size}, idf={idf.size}, expected={expected_dim}"
            )
        if panel.size and (
            np.any(panel < 0) or np.any(panel >= n_genes)
        ):
            raise ValueError("TF-IDF panel contains gene indices outside the vocabulary")
        if panel.size > 1 and np.any(panel[1:] <= panel[:-1]):
            raise ValueError("TF-IDF panel_gene_index must be strictly increasing")
        if not np.isfinite(idf).all() or np.any(idf <= 0):
            raise ValueError("TF-IDF idf values must be finite and positive")
        tf_scale = float(raw.get("tf_scale", 1.0e4))
        if not np.isfinite(tf_scale) or tf_scale <= 0:
            raise ValueError("TF-IDF tf_scale must be finite and positive")
        metadata = {
            str(key): value for key, value in raw.items()
            if key not in {"panel_gene_index", "idf"}
        }
        return cls(panel, idf, tf_scale=tf_scale, metadata=metadata)

    @property
    def dimension(self) -> int:
        return int(self.panel_gene_index.size)

    def transform(self, indexed_batch: dict[str, Any]) -> np.ndarray:
        """Materialize a ``[B, panel_dim]`` log-TF-IDF view from CSR tokens."""
        row_ptr = np.asarray(indexed_batch["row_ptr"], dtype=np.int64)
        gene_index = np.asarray(indexed_batch["gene_index"], dtype=np.int64)
        count = np.asarray(indexed_batch["count"], dtype=np.float32)
        if row_ptr.ndim != 1 or row_ptr.size < 2:
            raise ValueError("indexed batch row_ptr must contain B+1 entries")
        if gene_index.ndim != 1 or count.ndim != 1:
            raise ValueError("indexed batch gene_index/count must be one-dimensional")
        if gene_index.size != count.size or int(row_ptr[-1]) != count.size:
            raise ValueError("indexed batch CSR arrays are inconsistent")

        batch_size = int(row_ptr.size - 1)
        panel = self.panel_gene_index
        values = np.zeros((batch_size, self.dimension), dtype=np.float32)
        for row in range(batch_size):
            start, stop = int(row_ptr[row]), int(row_ptr[row + 1])
            if start == stop or not self.dimension:
                continue
            genes = gene_index[start:stop]
            positions = np.searchsorted(panel, genes)
            candidate = positions < self.dimension
            if np.any(candidate):
                candidate_positions = positions[candidate]
                matches = panel[candidate_positions] == genes[candidate]
                valid = np.zeros_like(candidate, dtype=bool)
                valid[candidate] = matches
                if np.any(valid):
                    # ``read_batch`` coalesces duplicate gene tokens, but
                    # add.at keeps this transform correct for hand-built CSR
                    # batches used by tests and diagnostics as well.
                    np.add.at(values[row], positions[valid], count[start:stop][valid])
        library = values.sum(axis=1, keepdims=True)
        values = values / np.clip(library, 1.0, None)
        values = np.log1p(values * self.idf[None, :] * self.tf_scale)
        if not np.isfinite(values).all():
            raise ValueError("TF-IDF transform produced non-finite values")
        return values.astype(np.float32, copy=False)


class MetadataTensorizer:
    def __init__(self, vocabulary_path: str | Path, config: ExperimentConfig):
        self.config = config
        self.vocab = json.loads(Path(vocabulary_path).read_text())
        expected = config.model.context.categorical_vocab_sizes
        actual = {name: len(vocab) for name, vocab in self.vocab["categorical"].items()}
        if expected != actual:
            raise ValueError(f"configuration/vocabulary size mismatch: {expected} != {actual}")
        self.tfidf_auxiliary = (
            TfidfAuxiliarySpec.from_path(
                config.data.optional_modality_path,
                expected_dim=config.model.encoder.optional_modality_dim,
                n_genes=config.model.n_genes,
                expected_fit_split=config.data.split,
                expected_fit_assay_types=config.data.assay_types,
            )
            if config.model.encoder.optional_modality_dim else None
        )

    @staticmethod
    def _value(value: Any) -> str:
        return MISSING if pd.isna(value) or value == "" else str(value)

    def _canonical_metadata(
        self, metadata: pd.DataFrame, *, require_species: bool
    ) -> pd.DataFrame:
        return canonicalize_metadata(
            metadata,
            species_vocab=self.vocab.get("species"),
            require_species=require_species,
        )

    def _categorical_values(self, metadata: pd.DataFrame) -> np.ndarray:
        encoded = []
        for field in self.config.model.context.categorical_vocab_sizes:
            vocab = self.vocab["categorical"][field]
            values = (
                metadata[field]
                if field in metadata
                else pd.Series([None] * len(metadata), index=metadata.index)
            )
            encoded.append(
                np.asarray(
                    [
                        vocab.get(self._value(value), vocab[UNKNOWN])
                        for value in values
                    ],
                    dtype=np.int64,
                )
            )
        return (
            np.stack(encoded, axis=1)
            if encoded
            else np.empty((len(metadata), 0), np.int64)
        )

    def categorical(self, metadata: pd.DataFrame) -> np.ndarray:
        canonical = self._canonical_metadata(metadata, require_species=False)
        return self._categorical_values(canonical)

    @staticmethod
    def _continuous_values(metadata: pd.DataFrame) -> np.ndarray:
        point = pd.to_numeric(metadata["time_point_days"], errors="coerce")
        lower = pd.to_numeric(metadata["time_lower_days"], errors="coerce")
        upper = pd.to_numeric(metadata["time_upper_days"], errors="coerce")
        return np.stack(
            [
                point.fillna(0.0).to_numpy(np.float32) / 365.2425,
                point.notna().to_numpy(np.float32),
                lower.fillna(0.0).to_numpy(np.float32) / 365.2425,
                upper.fillna(0.0).to_numpy(np.float32) / 365.2425,
            ],
            axis=1,
        )

    @staticmethod
    def continuous(metadata: pd.DataFrame) -> np.ndarray:
        canonical = canonicalize_metadata(metadata, require_species=False)
        return MetadataTensorizer._continuous_values(canonical)

    def _species_values(self, metadata: pd.DataFrame) -> np.ndarray:
        vocab = self.vocab["species"]
        values = [self._value(value) for value in metadata["species"]]
        unknown = sorted(set(values) - set(vocab))
        if unknown:
            raise ValueError(
                "species values are not present in the vocabulary: "
                + ", ".join(unknown)
            )
        return np.asarray([vocab[value] for value in values], dtype=np.int64)

    def species(self, metadata: pd.DataFrame) -> np.ndarray:
        canonical = self._canonical_metadata(metadata, require_species=True)
        return self._species_values(canonical)

    def tensorize(self, indexed_batch: dict[str, Any], metadata: pd.DataFrame,
                  device: torch.device | str) -> WorldModelBatch:
        metadata = self._canonical_metadata(metadata, require_species=True)
        expression = SparseExpressionBatch(
            row_ptr=torch.as_tensor(indexed_batch["row_ptr"], dtype=torch.long, device=device),
            gene_index=torch.as_tensor(indexed_batch["gene_index"], dtype=torch.long, device=device),
            count=torch.as_tensor(indexed_batch["count"], dtype=torch.float32, device=device),
            library_size=torch.as_tensor(indexed_batch["library_size"], dtype=torch.float32, device=device),
            measured_gene_index=torch.as_tensor(
                indexed_batch["measured_gene_index"], dtype=torch.long, device=device
            ),
            missing_token_mask=(
                torch.as_tensor(
                    indexed_batch["missing_token_mask"],
                    dtype=torch.bool,
                    device=device,
                )
                if "missing_token_mask" in indexed_batch
                else None
            ),
            measured_zero_mask=(
                torch.as_tensor(
                    indexed_batch["measured_zero_mask"],
                    dtype=torch.bool,
                    device=device,
                )
                if "measured_zero_mask" in indexed_batch
                else None
            ),
            panel_missing_summary=(
                torch.as_tensor(
                    indexed_batch["panel_missing_summary"],
                    dtype=torch.float32,
                    device=device,
                )
                if "panel_missing_summary" in indexed_batch
                else None
            ),
        )
        # Coordinates are an optional observation, not a value to impute.  A
        # molecular-only batch keeps ``spatial_coords=None``; a mixed batch
        # receives a safe placeholder tensor plus an explicit per-cell mask so
        # graph/niche modules can exclude unavailable rows from both message
        # sources and destinations.  Never pass NaN coordinates downstream.
        coordinate_frame = metadata[["spatial_x", "spatial_y"]]
        coordinate_presence = coordinate_frame.notna().all(axis=1).to_numpy(bool)
        if "spatial_presence" in metadata:
            declared = metadata["spatial_presence"]
            if declared.dtype == bool:
                declared_presence = declared.to_numpy(bool)
            else:
                numeric = pd.to_numeric(declared, errors="coerce")
                if numeric.isna().any() or ((numeric < 0) | (numeric > 1)).any():
                    raise ValueError(
                        "metadata.spatial_presence must be boolean or numeric in [0, 1]"
                    )
                declared_presence = numeric.to_numpy(np.float32) > 0.5
            # A declared present row without two finite coordinates is
            # ambiguous and must fail closed instead of silently becoming an
            # origin cell.
            if np.any(declared_presence & ~coordinate_presence):
                raise ValueError(
                    "spatial_presence marks rows present but coordinates are missing"
                )
            coordinate_presence &= declared_presence
        if np.any(coordinate_presence):
            coordinates = coordinate_frame.to_numpy(np.float32, copy=True)
            coordinates[~coordinate_presence] = 0.0
            spatial = torch.as_tensor(
                coordinates,
                dtype=torch.float32,
                device=device,
            )
            # Preserve the historical all-present representation (absence of
            # a mask means every coordinate row is usable); only mixed batches
            # need an explicit mask.  This keeps old callers/checkpoints
            # stable while making partial availability unambiguous.
            spatial_presence = (
                torch.as_tensor(coordinate_presence, dtype=torch.bool, device=device)
                if not bool(np.all(coordinate_presence))
                else None
            )
        else:
            spatial = None
            spatial_presence = None
        sample_codes, _ = pd.factorize(metadata["sample_key"], sort=True)
        point = pd.to_numeric(metadata["time_point_days"], errors="coerce").fillna(0.0)
        control_dim = self.config.model.intervention.control_dim
        control = torch.zeros((len(metadata), control_dim), device=device) if control_dim else None
        optional_modalities = None
        if self.tfidf_auxiliary is not None:
            optional_modalities = torch.as_tensor(
                self.tfidf_auxiliary.transform(indexed_batch),
                dtype=torch.float32,
                device=device,
            )
        return WorldModelBatch(
            expression=expression,
            categorical_context=torch.as_tensor(
                self._categorical_values(metadata), dtype=torch.long, device=device
            ),
            continuous_context=torch.as_tensor(
                self._continuous_values(metadata), dtype=torch.float32, device=device
            ),
            time_days=torch.as_tensor(point.to_numpy(np.float32), device=device),
            species_id=torch.as_tensor(
                self._species_values(metadata), dtype=torch.long, device=device
            ),
            batch_index=torch.as_tensor(sample_codes, dtype=torch.long, device=device),
            optional_modalities=optional_modalities,
            spatial_coords=spatial,
            spatial_presence=spatial_presence,
            control=control,
            delta_control=torch.zeros_like(control) if control is not None else None,
        )

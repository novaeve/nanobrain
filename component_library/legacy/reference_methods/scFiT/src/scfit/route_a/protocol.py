import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .data import _load_preprocessed_temporal_bundle, parse_time_value
from scfit.paths import as_posix_path


REQUIRED_PROTOCOL_FIELDS = {
    "protocol_name",
    "data_bundle_path",
    "eligible_time_bins",
    "observed_time_bins",
    "heldout_time_bins",
    "task_type",
    "source_policy",
    "eval_time_bins",
    "normalization",
    "hvg_dim",
}


def _time_sort_key(time_bin: str, numeric_map: Dict[str, float] = None):
    time_bin = str(time_bin)
    numeric = np.nan
    if numeric_map is not None:
        numeric = float(numeric_map.get(time_bin, np.nan))
    if not np.isfinite(numeric):
        numeric = float(parse_time_value(time_bin))
    if np.isfinite(numeric):
        return (0, numeric, time_bin)
    return (1, time_bin)


def _sorted_time_bins(time_bins: List[str], numeric_map: Dict[str, float] = None) -> List[str]:
    return sorted([str(t) for t in time_bins], key=lambda x: _time_sort_key(x, numeric_map=numeric_map))


def load_protocol_manifest(path: str) -> Dict:
    protocol_path = Path(as_posix_path(path))
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol file not found: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as f:
        if protocol_path.suffix.lower() in {".json"}:
            manifest = json.load(f)
        else:
            manifest = yaml.safe_load(f)
    missing = REQUIRED_PROTOCOL_FIELDS - set(manifest.keys())
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    manifest["data_bundle_path"] = as_posix_path(manifest["data_bundle_path"], base_dir=protocol_path.parent)
    _validate_protocol_manifest(manifest)
    return manifest


def _validate_protocol_manifest(manifest: Dict):
    eligible = [str(x) for x in manifest["eligible_time_bins"]]
    observed = [str(x) for x in manifest["observed_time_bins"]]
    heldout = [str(x) for x in manifest["heldout_time_bins"]]
    eval_bins = [str(x) for x in manifest["eval_time_bins"]]

    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible_time_bins contains duplicates")
    if len(set(observed)) != len(observed):
        raise ValueError("observed_time_bins contains duplicates")
    if len(set(heldout)) != len(heldout):
        raise ValueError("heldout_time_bins contains duplicates")
    if set(observed) & set(heldout):
        raise ValueError("observed_time_bins and heldout_time_bins must be disjoint")
    if not set(observed).issubset(set(eligible)):
        raise ValueError("observed_time_bins must be subset of eligible_time_bins")
    if not set(heldout).issubset(set(eligible)):
        raise ValueError("heldout_time_bins must be subset of eligible_time_bins")
    if not set(eval_bins).issubset(set(heldout)):
        raise ValueError("eval_time_bins must be subset of heldout_time_bins")
    if str(manifest.get("direction_mode", "") or manifest["task_type"]) not in {
        "forward_only",
        "bidirectional",
        "interpolation",
        "near_extrapolation",
        "far_extrapolation",
        "three_interpolation",
        "three_forecasting",
        "two_forecasting",
        "remove_recovery",
    }:
        raise ValueError("invalid direction_mode/task_type")
    display_time_mode = str(manifest.get("display_time_mode", "original"))
    if display_time_mode not in {"original", "index_0based", "index_1based"}:
        raise ValueError(f"invalid display_time_mode={display_time_mode}")
    if "display_time_labels" in manifest:
        labels = manifest["display_time_labels"] or {}
        if not isinstance(labels, dict):
            raise ValueError("display_time_labels must be a mapping")
        eligible = set(str(x) for x in manifest["eligible_time_bins"])
        label_keys = set(str(x) for x in labels.keys())
        if not eligible.issubset(label_keys):
            missing = sorted(eligible - label_keys)
            raise ValueError(f"display_time_labels missing eligible bins: {missing}")


def _ensure_obs_name_column(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if "obs_name" in metadata_df.columns:
        return metadata_df
    if metadata_df.index.name == "obs_name":
        return metadata_df.reset_index()
    normalized = metadata_df.copy()
    normalized.insert(0, "obs_name", normalized.index.to_numpy().astype(str))
    return normalized


def _build_time_index(metadata_df: pd.DataFrame, time_col: str = "author_day") -> Dict[str, np.ndarray]:
    series = metadata_df[time_col].astype(str)
    return {
        time_bin: np.flatnonzero(series.to_numpy() == time_bin).astype(np.int64)
        for time_bin in _sorted_time_bins(series.unique().tolist())
    }


def _build_time_numeric_map(
    metadata_df: pd.DataFrame,
    time_col: str = "author_day",
    numeric_col: str = "author_day_numeric",
) -> Dict[str, float]:
    labels = metadata_df[time_col].astype(str).to_numpy()
    numeric_values = metadata_df[numeric_col].to_numpy(dtype=np.float32) if numeric_col in metadata_df.columns else None
    numeric_map = {}
    for time_bin in pd.unique(labels).tolist():
        if numeric_values is not None:
            mask = labels == str(time_bin)
            vals = numeric_values[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                numeric_map[str(time_bin)] = float(np.median(vals))
                continue
        parsed = float(parse_time_value(time_bin))
        numeric_map[str(time_bin)] = parsed
    return numeric_map


def _neighbor_maps(ordered_bins: List[str], observed_bins: List[str], numeric_map: Dict[str, float] = None):
    observed_sorted = _sorted_time_bins(observed_bins, numeric_map=numeric_map)
    observed_numeric = np.asarray(
        [
            numeric_map.get(t, parse_time_value(t)) if numeric_map is not None else parse_time_value(t)
            for t in observed_sorted
        ],
        dtype=np.float32,
    )
    prev_map = {}
    next_map = {}
    for time_bin in ordered_bins:
        t = float(numeric_map.get(time_bin, parse_time_value(time_bin))) if numeric_map is not None else float(parse_time_value(time_bin))
        if not np.isfinite(t):
            prev_map[time_bin] = None
            next_map[time_bin] = None
            continue
        prev_candidates = [obs for obs, obs_t in zip(observed_sorted, observed_numeric) if np.isfinite(obs_t) and obs_t < t]
        next_candidates = [obs for obs, obs_t in zip(observed_sorted, observed_numeric) if np.isfinite(obs_t) and obs_t > t]
        prev_map[time_bin] = prev_candidates[-1] if prev_candidates else None
        next_map[time_bin] = next_candidates[0] if next_candidates else None
    return prev_map, next_map


def _build_time_display_map(ordered_bins: List[str], manifest: Dict) -> Dict[str, str]:
    custom = manifest.get("display_time_labels")
    if custom:
        return {str(k): str(v) for k, v in custom.items()}

    mode = str(manifest.get("display_time_mode", "original"))
    if mode == "original":
        return {str(time_bin): str(time_bin) for time_bin in ordered_bins}
    if mode == "index_0based":
        return {str(time_bin): f"t={idx}" for idx, time_bin in enumerate(ordered_bins)}
    if mode == "index_1based":
        return {str(time_bin): f"t={idx + 1}" for idx, time_bin in enumerate(ordered_bins)}
    raise ValueError(f"unsupported display_time_mode={mode}")


def load_protocol_bundle(protocol_path: str) -> Dict:
    manifest = load_protocol_manifest(protocol_path)
    expr, metadata_df, bundle_meta = _load_preprocessed_temporal_bundle(manifest["data_bundle_path"])
    metadata_df = _ensure_obs_name_column(metadata_df)

    if int(bundle_meta["fixed_panel_size"]) != int(manifest["hvg_dim"]):
        raise ValueError(
            f"hvg_dim mismatch: protocol={manifest['hvg_dim']} bundle={bundle_meta['fixed_panel_size']}"
        )

    time_to_indices = _build_time_index(metadata_df, time_col="author_day")
    time_numeric_map = _build_time_numeric_map(metadata_df, time_col="author_day", numeric_col="author_day_numeric")
    missing_bins = set(manifest["eligible_time_bins"]) - set(time_to_indices.keys())
    if missing_bins:
        raise ValueError(f"protocol references missing time bins: {sorted(missing_bins)}")

    eligible_bins = _sorted_time_bins(manifest["eligible_time_bins"], numeric_map=time_numeric_map)
    observed_bins = _sorted_time_bins(manifest["observed_time_bins"], numeric_map=time_numeric_map)
    heldout_bins = _sorted_time_bins(manifest["heldout_time_bins"], numeric_map=time_numeric_map)
    eval_bins = _sorted_time_bins(manifest["eval_time_bins"], numeric_map=time_numeric_map)

    eligible_counts = {t: int(len(time_to_indices[t])) for t in eligible_bins}
    min_cells_per_time = manifest.get("min_cells_per_time")
    if min_cells_per_time is not None:
        bad_bins = [t for t in eligible_bins if eligible_counts[t] < int(min_cells_per_time)]
        if bad_bins:
            raise ValueError(
                "protocol contains bins below min_cells_per_time "
                f"({int(min_cells_per_time)}): {bad_bins}"
            )
    min_time_numeric = manifest.get("min_time_numeric")
    if min_time_numeric is not None:
        bad_bins = [t for t in eligible_bins if float(time_numeric_map.get(t, np.nan)) < float(min_time_numeric)]
        if bad_bins:
            raise ValueError(
                "protocol contains bins below min_time_numeric "
                f"({float(min_time_numeric)}): {bad_bins}"
            )
    prev_map, next_map = _neighbor_maps(eligible_bins, observed_bins, numeric_map=time_numeric_map)
    time_display_map = _build_time_display_map(eligible_bins, manifest)

    if any((bundle_t is None) for bundle_t in [metadata_df.get("author_day"), metadata_df.get("author_day_numeric")]):
        raise ValueError(
            "metadata_df must contain `author_day` and `author_day_numeric` for Route A protocols"
        )

    cell_type_col = "author_cell_type" if "author_cell_type" in metadata_df.columns else None
    cell_type_names = []
    if cell_type_col is not None:
        cell_type_names = sorted(metadata_df[cell_type_col].astype(str).unique().tolist())

    return {
        "manifest": manifest,
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_name": str(manifest["protocol_name"]),
        "expression": expr,
        "metadata_df": metadata_df,
        "bundle_meta": bundle_meta,
        "ordered_time_bins": eligible_bins,
        "observed_time_bins": observed_bins,
        "heldout_time_bins": heldout_bins,
        "eval_time_bins": eval_bins,
        "time_to_indices": {k: time_to_indices[k] for k in eligible_bins},
        "time_numeric_map": {k: float(time_numeric_map[k]) for k in eligible_bins},
        "time_display_map": {k: str(time_display_map[k]) for k in eligible_bins},
        "eligible_counts": eligible_counts,
        "prev_observed_time": prev_map,
        "next_observed_time": next_map,
        "direction_mode": str(manifest.get("direction_mode", "forward_only")),
        "task_type": str(manifest["task_type"]),
        "source_policy": str(manifest["source_policy"]),
        "cell_type_col": cell_type_col,
        "cell_type_names": cell_type_names,
    }


def build_scnode_inputs(bundle: Dict):
    ordered_bins = bundle["ordered_time_bins"]
    expr = bundle["expression"]
    traj_data = [np.asarray(expr[bundle["time_to_indices"][time_bin]], dtype=np.float32) for time_bin in ordered_bins]
    observed = set(bundle["observed_time_bins"])
    heldout = set(bundle["heldout_time_bins"])
    train_tps = [i for i, time_bin in enumerate(ordered_bins) if time_bin in observed]
    test_tps = [i for i, time_bin in enumerate(ordered_bins) if time_bin in heldout]
    return {
        "traj_data": traj_data,
        "ordered_time_bins": ordered_bins,
        "train_tps": train_tps,
        "test_tps": test_tps,
        "all_tps": list(range(len(ordered_bins))),
    }


def resolve_generation_sources(bundle: Dict, target_time_bin: str):
    target_time_bin = str(target_time_bin)
    policy = bundle["source_policy"]
    prev_time = bundle["prev_observed_time"].get(target_time_bin)
    next_time = bundle["next_observed_time"].get(target_time_bin)
    numeric_map = bundle.get("time_numeric_map", {}) or {}

    def _numeric_time(time_bin: str) -> float:
        value = float(numeric_map.get(str(time_bin), np.nan))
        if not np.isfinite(value):
            value = float(parse_time_value(time_bin))
        return value

    target_numeric = _numeric_time(target_time_bin)
    if not np.isfinite(target_numeric):
        raise ValueError(f"failed to resolve numeric time for target={target_time_bin}")

    if policy == "previous_observed":
        if prev_time is None:
            raise ValueError(f"no previous observed time for target={target_time_bin}")
        return [{"time_bin": prev_time, "weight": 1.0}]

    if policy == "bidirectional_linear":
        if prev_time is None or next_time is None:
            raise ValueError(f"bidirectional source policy requires both neighbors for {target_time_bin}")
        prev_numeric = _numeric_time(prev_time)
        next_numeric = _numeric_time(next_time)
        if not np.isfinite(prev_numeric) or not np.isfinite(next_numeric):
            raise ValueError(
                f"failed to resolve numeric time for interpolation neighbors: "
                f"target={target_time_bin}, prev={prev_time}, next={next_time}"
            )
        interval = max(next_numeric - prev_numeric, 1e-6)
        prev_weight = float((next_numeric - target_numeric) / interval)
        next_weight = float((target_numeric - prev_numeric) / interval)
        total = max(prev_weight + next_weight, 1e-8)
        sources = [
            {"time_bin": prev_time, "weight": prev_weight / total},
            {"time_bin": next_time, "weight": next_weight / total},
        ]
        for source in sources:
            weight = float(source["weight"])
            if not np.isfinite(weight):
                raise ValueError(
                    f"invalid interpolation weight for target={target_time_bin}: "
                    f"{source['time_bin']} -> {weight}"
                )
        return sources

    if policy == "hybrid_observed":
        if prev_time is not None and next_time is not None:
            prev_numeric = _numeric_time(prev_time)
            next_numeric = _numeric_time(next_time)
            if not np.isfinite(prev_numeric) or not np.isfinite(next_numeric):
                raise ValueError(
                    f"failed to resolve numeric time for hybrid neighbors: "
                    f"target={target_time_bin}, prev={prev_time}, next={next_time}"
                )
            interval = max(next_numeric - prev_numeric, 1e-6)
            prev_weight = float((next_numeric - target_numeric) / interval)
            next_weight = float((target_numeric - prev_numeric) / interval)
            total = max(prev_weight + next_weight, 1e-8)
            return [
                {"time_bin": prev_time, "weight": prev_weight / total},
                {"time_bin": next_time, "weight": next_weight / total},
            ]
        if prev_time is not None:
            return [{"time_bin": prev_time, "weight": 1.0}]
        if next_time is not None:
            return [{"time_bin": next_time, "weight": 1.0}]
        raise ValueError(f"hybrid_observed could not find any observed neighbor for target={target_time_bin}")

    raise ValueError(f"unsupported source_policy={policy}")

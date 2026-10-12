"""Small, backed H5AD helpers used without loading complete expression matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in values
        ], dtype=object)
    return values


def read_column(group: h5py.Group, name: str,
                indices: np.ndarray | None = None) -> np.ndarray:
    """Read an AnnData dataframe column, including categorical encoding."""
    node = group[name]
    # A few exported H5AD files omit the AnnData categorical attribute while
    # retaining the canonical ``categories``/``codes`` group layout. Treat
    # that layout as categorical as well; otherwise a backed read attempts to
    # slice the group itself and fails before expression is touched.
    is_categorical = (
        isinstance(node, h5py.Group)
        and (
            node.attrs.get("encoding-type") == "categorical"
            or {"categories", "codes"}.issubset(node.keys())
        )
    )
    if is_categorical:
        codes = np.asarray(node["codes"][:] if indices is None else node["codes"][indices])
        categories = decode_strings(node["categories"][:])
        out = np.empty(codes.shape, dtype=object)
        valid = codes >= 0
        out[~valid] = None
        out[valid] = categories[codes[valid]]
        return out
    values = node[:] if indices is None else node[indices]
    return decode_strings(np.asarray(values))


def dataframe_index_name(group: h5py.Group) -> str:
    value = group.attrs.get("_index", "_index")
    return value.decode() if isinstance(value, bytes) else str(value)


def matrix_shape(node: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    shape = node.attrs.get("shape", getattr(node, "shape", None))
    return int(shape[0]), int(shape[1])


def matrix_encoding(node: h5py.Group | h5py.Dataset) -> str:
    value = node.attrs.get("encoding-type", "array")
    return value.decode() if isinstance(value, bytes) else str(value)


def file_fingerprint(path: str | Path) -> str:
    """Cheap identity fingerprint; does not hash hundreds of GB of payload."""
    import hashlib
    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()


def existing_columns(group: h5py.Group, names: Iterable[str]) -> list[str]:
    return [name for name in names if name in group]

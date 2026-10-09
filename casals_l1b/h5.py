"""Small shared HDF5 access helpers for CASALS L1B files."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import warnings

DEFAULT_ATTR_KEYS = (
    "n_pulses",
    "n_sweeps",
    "n_tracks",
    "n_rx_bins",
    "n_tx_bins",
    "start_utca",
    "end_utca",
)


def normalize_attribute(value: Any) -> Any:
    """Convert one HDF5 attribute to ordinary Python values for metadata."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_global_attributes(h5: h5py.File) -> dict[str, Any]:
    return {str(key): normalize_attribute(value) for key, value in h5.attrs.items()}


def require_root_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    """Require the root-level dataset used by the original refh workflows."""
    if name not in h5 or not isinstance(h5[name], h5py.Dataset):
        raise KeyError(f"Required root dataset {name!r} was not found in the H5 file.")
    return h5[name]


def read_root_1d(h5: h5py.File, name: str) -> np.ndarray:
    values = np.asarray(require_root_dataset(h5, name)[...])
    if values.ndim != 1:
        raise ValueError(f"Dataset {name!r} is expected to be 1D, got shape {values.shape}.")
    return values


def find_dataset(h5: h5py.File, name: str) -> h5py.Dataset | None:
    """Find a root dataset or a unique recursive basename match."""
    normalized = name.strip("/")
    if normalized in h5 and isinstance(h5[normalized], h5py.Dataset):
        return h5[normalized]

    matches: list[str] = []

    def visitor(path: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and path.split("/")[-1] == normalized:
            matches.append(path)

    h5.visititems(visitor)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple datasets matched basename {name!r}: {matches}")
    return h5[matches[0]]


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    dataset = find_dataset(h5, name)
    if dataset is None:
        raise KeyError(f"Required dataset {name!r} was not found in the H5 file.")
    return dataset


def read_optional_1d(
    h5: h5py.File,
    name: str,
    n_expected: int,
    dtype: Any = np.float64,
) -> np.ndarray | None:
    return read_optional_array(
        h5, name, n_expected, dtype=dtype, ignore_size_mismatch=False
    )


def read_optional_array(
    h5: h5py.File,
    name: str,
    n_expected: int,
    dtype: Any = None,
    *,
    ignore_size_mismatch: bool = True,
) -> np.ndarray | None:
    dataset = find_dataset(h5, name)
    if dataset is None:
        return None
    values = np.asarray(dataset[...], dtype=dtype).reshape(-1)
    if values.size != int(n_expected):
        if ignore_size_mismatch:
            warnings.warn(
                f"Optional dataset {name!r} has size {values.size}, expected {int(n_expected)}; ignored."
            )
            return None
        raise ValueError(
            f"Optional dataset {name!r} has size {values.size}, expected {int(n_expected)}."
        )
    return values


def read_attrs_subset(
    h5: h5py.File,
    keys: Sequence[str] = DEFAULT_ATTR_KEYS,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in keys:
        if key not in h5.attrs:
            continue
        value = h5.attrs[key]
        if isinstance(value, bytes):
            attrs[key] = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            attrs[key] = value.tolist()
        elif hasattr(value, "item"):
            try:
                attrs[key] = value.item()
            except Exception:
                attrs[key] = str(value)
        else:
            attrs[key] = value
    return attrs


def find_dataset_path(
    h5: h5py.File,
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    """Return the first present candidate path, preserving candidate order."""
    for name in candidates:
        if name in h5:
            return name
    if required:
        raise KeyError(f"Could not find any dataset from candidates: {list(candidates)}")
    return None


def read_optional_dataset(
    h5: h5py.File,
    candidates: Sequence[str],
    n_expected: int,
) -> tuple[np.ndarray | None, str | None]:
    path = find_dataset_path(h5, candidates, required=False)
    if path is None:
        return None, None
    values = np.asarray(h5[path][...]).reshape(-1)
    if values.size != n_expected:
        raise ValueError(
            f"Optional dataset {path} has size {values.size}, expected {n_expected}"
        )
    return values, path


def scalar_attr_to_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if np.isscalar(value):
        return str(value)
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return None


def iter_scalar_attr_strings(h5: h5py.File):
    for key, value in h5.attrs.items():
        text = scalar_attr_to_text(value)
        if text is not None:
            yield f"/@{key}", text

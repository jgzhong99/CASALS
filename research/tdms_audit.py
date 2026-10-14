"""Small, read-only helpers for auditing CASALS L0 TDMS files."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np


CHANNEL_PATTERN = re.compile(r"^Sweep(?P<sweep>\d+)Step(?P<step>\d+)$", re.IGNORECASE)
_UNIT_PATTERN = re.compile(r"[\[(]\s*(nm|um|µm|μm|angstrom|hz|khz|mhz|ghz|s|ms|us|µs|μs|ns|tick|ticks|v|mv|count|counts)\s*[\])]", re.IGNORECASE)
_SCAN_TERMS = {
    "wvl", "wavelength", "lambda", "laser", "tuning", "step", "steps", "shift",
    "frequency", "calibration", "calibrated", "grating", "steering", "voltage",
    "sweep", "adc", "digitizer", "sample", "sampling", "timing", "pulse",
    "tx", "rx", "window", "acquisition", "detector", "tick",
}
_WAVELENGTH_TERMS = {
    "wvl", "wavelength", "lambda", "laser", "tuning", "frequency", "calibration",
    "grating", "steering", "voltage", "nm", "um", "µm", "μm", "angstrom",
}
_VALUE_TERMS = re.compile(
    r"(?<![a-z0-9])(wavelength|lambda|wvl|laser|tuning|frequency|calibration|grating|steering|voltage|shift.?cycle)(?![a-z0-9])",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, bytes):
        return {"encoding": "base64", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _tokens(name: str) -> set[str]:
    lowered = name.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    # TDMS keys often concatenate acronyms and words, for example WVLSteps/Sweep.
    for term in _SCAN_TERMS | _WAVELENGTH_TERMS:
        if term not in {"nm", "um", "µm", "μm", "v", "s", "hz"} and term in lowered:
            tokens.add(term)
    return tokens


def _unit(name: str) -> str | None:
    match = _UNIT_PATTERN.search(name)
    if match:
        return match.group(1).lower().replace("μ", "µ")
    suffix = re.search(r"(?:^|[_\s-])(nm|um|µm|μm|angstrom|hz|khz|mhz|ghz|s|ms|us|µs|μs|ns|tick|ticks|v|mv|count|counts)$", name, re.IGNORECASE)
    return suffix.group(1).lower().replace("μ", "µ") if suffix else None


def _property_rows(scope: str, properties: dict[str, Any], group_name: str | None = None,
                   channel_name: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for name, value in properties.items():
        raw = _jsonable(value)
        rows.append({
            "scope": scope,
            "property_name": str(name),
            "raw_data_type": type(value).__name__,
            "raw_value": raw,
            "readable_value": repr(raw),
            "unit": _unit(str(name)),
            "group_name": group_name,
            "channel_name": channel_name,
        })
    return rows


def discover_tdms_files(directory: str | Path) -> list[Path]:
    """Return real TDMS files recursively; index sidecars are excluded by suffix."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".tdms")


def select_tdms_file(project_root: str | Path, data_dir: str | Path,
                     tdms_path: str | Path | None = None) -> Path:
    """Resolve an explicit TDMS path, or accept automatic selection only when unique."""
    root = Path(project_root).resolve()
    if tdms_path is not None and str(tdms_path).strip():
        candidate = Path(tdms_path).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() != ".tdms":
            raise ValueError(f"Expected a .tdms file, not an index or other file: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(f"TDMS file does not exist: {candidate}")
        return candidate

    data_root = Path(data_dir)
    data_root = data_root if data_root.is_absolute() else root / data_root
    candidates = discover_tdms_files(data_root)
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No .tdms files were found below {data_root.resolve()}")
    listing = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise ValueError(
        f"Found {len(candidates)} TDMS files; set TDMS_PATH to choose one explicitly:\n{listing}"
    )


class TdmsReadError(OSError):
    """A TDMS and, when present, its index sidecar could not be read."""


@contextmanager
def open_tdms_metadata(path: str | Path) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Open metadata lazily, falling back to the data file when its index is invalid."""
    from nptdms import TdmsFile

    path = Path(path)
    index_path = Path(f"{path}_index")
    tdms = None
    raw_file = None
    index_error = None
    if index_path.is_file():
        try:
            tdms = TdmsFile.open(path)
            metadata_source = "tdms_index"
        except (EOFError, OSError, ValueError) as exc:
            index_error = f"{type(exc).__name__}: {exc}"
        if tdms is not None and not getattr(getattr(tdms, "_reader", None), "_segments", None):
            tdms.close()
            tdms = None
            index_error = "Index sidecar contained no readable TDMS segments."

    if tdms is None:
        try:
            if index_error is None:
                tdms = TdmsFile.open(path)
                metadata_source = "tdms"
            else:
                raw_file = path.open("rb")
                tdms = TdmsFile.open(raw_file)
                metadata_source = "tdms_raw_fallback"
        except Exception as exc:
            if raw_file is not None:
                raw_file.close()
            detail = f"TDMS read failed: {type(exc).__name__}: {exc}"
            if index_error:
                detail = f"Index sidecar read failed ({index_error}); raw TDMS read failed ({detail})."
            raise TdmsReadError(detail) from exc
        if not getattr(getattr(tdms, "_reader", None), "_segments", None):
            tdms.close()
            if raw_file is not None:
                raw_file.close()
            raise TdmsReadError("TDMS file contained no readable segments.")

    info = {"metadata_source": metadata_source, "index_error": index_error}
    try:
        yield tdms, info
    finally:
        tdms.close()
        if raw_file is not None:
            raw_file.close()


def metadata_inventory(tdms: Any) -> dict[str, Any]:
    """Capture current file/group/channel properties, shapes, and segment status."""
    file_properties = dict(tdms.properties)
    groups = []
    property_rows = _property_rows("file", file_properties)
    for group in tdms.groups():
        group_properties = dict(group.properties)
        group_item = {"name": group.name, "properties": _jsonable(group_properties), "channels": []}
        property_rows.extend(_property_rows("group", group_properties, group.name))
        for channel in group.channels():
            channel_properties = dict(channel.properties)
            group_item["channels"].append({
                "name": channel.name,
                "length": int(len(channel)),
                "data_type": str(channel.data_type),
                "dtype": str(channel.dtype),
                "properties": _jsonable(channel_properties),
            })
            property_rows.extend(_property_rows("channel", channel_properties, group.name, channel.name))
        groups.append(group_item)
    reader = getattr(tdms, "_reader", None)
    segments = getattr(reader, "_segments", None)
    file_status = getattr(tdms, "file_status", None)
    channel_statuses = getattr(file_status, "channel_statuses", None)
    try:
        channel_status_count = len(channel_statuses)
    except TypeError:
        channel_status_count = None
    channel_status_records = []
    if isinstance(channel_statuses, dict):
        status_items = channel_statuses.items()
    elif channel_statuses is not None:
        status_items = enumerate(channel_statuses)
    else:
        status_items = ()
    for channel_path, status in status_items:
        expected_length = getattr(status, "expected_length", None)
        read_length = getattr(status, "read_length", None)
        try:
            expected_length = int(expected_length) if expected_length is not None else None
        except (TypeError, ValueError, OverflowError):
            expected_length = None
        try:
            read_length = int(read_length) if read_length is not None else None
        except (TypeError, ValueError, OverflowError):
            read_length = None
        channel_status_records.append({
            "channel_path": str(channel_path),
            "expected_length": expected_length,
            "read_length": read_length,
            "shortfall_samples": (
                max(0, expected_length - read_length)
                if expected_length is not None and read_length is not None else None
            ),
        })
    status_mismatches = [
        row for row in channel_status_records
        if row["expected_length"] is not None and row["read_length"] is not None
        and row["expected_length"] != row["read_length"]
    ]
    return {
        "file_properties": _jsonable(file_properties),
        "groups": groups,
        "property_rows": property_rows,
        "reader_status": {
            "segment_count": len(segments) if segments is not None else None,
            "segment_count_source": "nptdms._reader._segments" if segments is not None else "unavailable",
            "segment_property_history": "merged_final_properties_only",
            "incomplete_final_segment": getattr(file_status, "incomplete_final_segment", None),
            "channel_status_count": channel_status_count,
            "channel_status_mismatch_count": (
                len(status_mismatches) if channel_statuses is not None else None
            ),
            "channel_status_shortfall_samples": (
                sum(row["shortfall_samples"] or 0 for row in channel_status_records)
                if channel_statuses is not None else None
            ),
            "channel_status_records": channel_status_records,
        },
    }


def scan_property_rows(property_rows: list[dict[str, Any]], *, wavelength_only: bool = False) -> list[dict[str, Any]]:
    """Select metadata fields by normalized keyword tokens, retaining the full inventory separately."""
    terms = _WAVELENGTH_TERMS if wavelength_only else _SCAN_TERMS
    return [row for row in property_rows if _tokens(row["property_name"]) & terms]


def _numeric_vector(value: Any) -> np.ndarray | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    try:
        values = np.asarray(value)
        if values.ndim == 0 or values.dtype.kind not in "iuf":
            values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim == 0 or values.size == 0 or values.dtype.kind not in "iuf":
        return None
    values = values.astype(np.float64, copy=False).reshape(-1)
    return values if np.all(np.isfinite(values)) else None


def metadata_value_candidates(property_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find wavelength-related names/strings and retain small unlabeled numeric arrays for review."""
    candidates = []
    for row in property_rows:
        value = row["raw_value"]
        reasons = []
        if _tokens(row["property_name"]) & _WAVELENGTH_TERMS:
            reasons.append("property_name")
        value_terms = {term.lower() for term in _VALUE_TERMS.findall(value)} if isinstance(value, str) else set()
        if value_terms:
            reasons.append("string_value")
        values = _numeric_vector(value)
        if values is not None and values.size <= 4096:
            reasons.append("numeric_array" if not isinstance(value, str) else "serialized_numeric_array")
        if not reasons:
            continue
        candidate = {**row, "candidate_reasons": reasons, "value_terms": sorted(value_terms, key=str.lower)}
        if values is not None:
            candidate["numeric_value_count"] = int(values.size)
            candidate["numeric_value_min"] = float(values.min())
            candidate["numeric_value_max"] = float(values.max())
        candidates.append(candidate)
    return candidates


def channel_property_variation(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in inventory["property_rows"]:
        if row["scope"] == "channel":
            by_name.setdefault(row["property_name"], []).append(row)
    return [{
        "property_name": name,
        "channel_count": len(rows),
        "distinct_value_count": len({json.dumps(row["raw_value"], sort_keys=True, ensure_ascii=False) for row in rows}),
        "values_by_channel": {row["channel_name"]: row["raw_value"] for row in rows},
    } for name, rows in sorted(by_name.items())]


def _property_value(properties: dict[str, Any], normalized_name: str) -> Any:
    for key, value in properties.items():
        if re.sub(r"[^a-z0-9]", "", str(key).lower()) == normalized_name:
            return value
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return result if result > 0 else None


def analyze_channel_layout(inventory: dict[str, Any], group_name: str | None = None) -> dict[str, Any]:
    """Parse Sweep{s}Step{st}, compare with declared dimensions, and derive viewer order."""
    groups = inventory["groups"]
    if not groups:
        raise ValueError("TDMS file has no groups.")
    if group_name is not None:
        group = next((item for item in groups if item["name"] == group_name), None)
        if group is None:
            raise ValueError(f"TDMS group does not exist: {group_name}")
    else:
        group = next((item for item in groups if item["name"] == "CASALS"), groups[0])
    properties = group["properties"]
    steps = _positive_int(_property_value(properties, "wvlstepssweep"))
    sweeps = _positive_int(_property_value(properties, "sweepsshiftcycle"))
    rows_per_file = _positive_int(_property_value(properties, "sweepsfile"))

    parsed = []
    observed_keys: dict[tuple[int, int], list[dict[str, Any]]] = {}
    all_names = [item["name"] for item in group["channels"]]
    duplicate_names = sorted(name for name in set(all_names) if all_names.count(name) > 1)
    for file_order, channel in enumerate(group["channels"]):
        match = CHANNEL_PATTERN.fullmatch(channel["name"])
        if match is None:
            parsed.append({**channel, "group_name": group["name"], "file_order": file_order,
                           "sweep": None, "step": None, "footprint_index": None,
                           "layout_status": "unrecognized_channel_name"})
            continue
        sweep, step = int(match.group("sweep")), int(match.group("step"))
        item = {**channel, "group_name": group["name"], "file_order": file_order,
                "sweep": sweep, "step": step, "footprint_index": None, "layout_status": "parsed"}
        parsed.append(item)
        observed_keys.setdefault((sweep, step), []).append(item)

    expected = {(sweep, step) for sweep in range(sweeps) for step in range(steps)} if steps and sweeps else set()
    duplicates = sorted(key for key, items in observed_keys.items() if len(items) > 1)
    for item in parsed:
        if item["sweep"] is None:
            continue
        key = (item["sweep"], item["step"])
        if key in duplicates:
            item["layout_status"] = "duplicate_step_sweep"
        elif expected and key not in expected:
            item["layout_status"] = "extra_step_sweep"
        elif expected:
            item["layout_status"] = "ok"
        if steps and sweeps and item["layout_status"] in {"ok", "parsed"}:
            item["footprint_index"] = item["step"] * sweeps + item["sweep"]

    missing = sorted(expected - set(observed_keys))
    recognized_count = sum(item["sweep"] is not None for item in parsed)
    file_order_keys = [(item["sweep"], item["step"]) for item in parsed if item["sweep"] is not None]
    sweep_major_order = [(sweep, step) for sweep in range(sweeps or 0) for step in range(steps or 0)]
    step_major_order = [(sweep, step) for step in range(steps or 0) for sweep in range(sweeps or 0)]
    file_order_is_sweep_major = bool(expected) and file_order_keys == sweep_major_order
    file_order_is_step_major = bool(expected) and file_order_keys == step_major_order
    lengths = sorted({int(item["length"]) for item in parsed})
    channel_lengths_consistent = len(lengths) <= 1
    samples_per_row = None
    if rows_per_file and len(lengths) == 1 and lengths[0] % rows_per_file == 0:
        samples_per_row = lengths[0] // rows_per_file
    row_count_matches = bool(rows_per_file and all(length % rows_per_file == 0 for length in lengths))
    complete = bool(
        expected and not missing and not duplicates and not duplicate_names
        and recognized_count == len(expected) and all(item["layout_status"] == "ok" for item in parsed)
        and channel_lengths_consistent and row_count_matches
    )
    return {
        "group_name": group["name"],
        "steps_per_sweep": steps,
        "sweeps_per_cycle": sweeps,
        "footprints_per_row": steps * sweeps if steps and sweeps else None,
        "rows_per_file": rows_per_file,
        "expected_channel_count": len(expected) if expected else None,
        "actual_channel_count": len(group["channels"]),
        "recognized_channel_count": recognized_count,
        "missing_step_sweep": [list(key) for key in missing],
        "duplicate_step_sweep": [list(key) for key in duplicates],
        "duplicate_channel_names": duplicate_names,
        "unrecognized_channels": [item["name"] for item in parsed if item["sweep"] is None],
        "extra_step_sweep": [
            [item["sweep"], item["step"]] for item in parsed if item["layout_status"] == "extra_step_sweep"
        ],
        "channel_order_in_file": (
            "sweep-major then step" if file_order_is_sweep_major else
            "step-major then sweep" if file_order_is_step_major else "other or incomplete"
        ),
        "file_order_matches_sweep_major_then_step": file_order_is_sweep_major,
        "file_order_matches_step_major_then_sweep": file_order_is_step_major,
        "channel_lengths": lengths,
        "channel_lengths_consistent": channel_lengths_consistent,
        "rows_file_divide_channel_length": row_count_matches,
        "samples_per_row": samples_per_row,
        "layout_complete": complete,
        "channels": parsed,
    }


def _unit_to_nm(unit: str | None) -> float | None:
    if unit == "nm":
        return 1.0
    if unit in {"um", "\u00b5m", "\u03bcm"}:
        return 1000.0
    if unit == "angstrom":
        return 0.1
    return None


def _finite_number(row: dict[str, Any]) -> float | None:
    value = row["raw_value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    return float(value)


def _source(row: dict[str, Any], source_file: str | None) -> str:
    owner = row.get("group_name") or row.get("channel_name") or "file"
    prefix = f"{source_file}:" if source_file else ""
    return f"{prefix}{row['scope']}:{owner}.{row['property_name']}"


def _mapping_row(
    channel: dict[str, Any], wavelength_nm: float, source: str, calibration_status: str,
) -> dict[str, Any]:
    return {
        "channel_name": channel["name"],
        "step": channel["step"],
        "sweep": channel["sweep"],
        "footprint_index": channel["footprint_index"],
        "wavelength_nm": float(wavelength_nm),
        "source": source,
        "calibration_status": calibration_status,
    }


def _lookup_order(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z]+", " ", value.lower()).strip()
    if "step major" in normalized and "sweep minor" in normalized:
        return "step_major"
    if "sweep major" in normalized and "step minor" in normalized:
        return "sweep_major"
    if "footprint index" in normalized:
        return "footprint_index"
    return None


def _parameter_formula_is_complete(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    expression, _, applicability = value.lower().partition(";")
    expression = re.sub(r"^(?:wavelength|lambda)\s*=\s*", "", expression.strip())
    expression = re.sub(r"[\s()]", "", expression)
    supported = expression in {
        "start+step*increment+sweep*sweep_offset",
        "start+increment*step+sweep*sweep_offset",
    }
    repeats = bool(re.search(r"repeat\w*\s+(?:each|every)\s+shift.?cycles?", applicability))
    return supported and repeats


def wavelength_evidence(
    inventory: dict[str, Any], layout: dict[str, Any], *, source_file: str | None = None,
) -> dict[str, Any]:
    # Emit a physical map only when units, all indices, and the mapping rule are explicit.
    properties = inventory["property_rows"]
    candidates = metadata_value_candidates(properties)
    physical_terms = {"wvl", "wavelength", "lambda"}
    ordered_channels = sorted(
        (item for item in layout["channels"] if item["footprint_index"] is not None),
        key=lambda item: item["footprint_index"],
    )
    conflicts: list[str] = []
    partial_direct: dict[str, float] = {}
    direct_mapping: list[dict[str, Any]] = []
    direct_complete = bool(layout["layout_complete"] and ordered_channels)

    for channel in ordered_channels:
        rows = [
            row for row in properties
            if row["scope"] == "channel" and row["channel_name"] == channel["name"]
            and _tokens(row["property_name"]) & physical_terms
        ]
        physical_values = []
        for row in rows:
            scale = _unit_to_nm(row["unit"])
            value = _finite_number(row)
            if scale is not None and value is not None:
                physical_values.append((value * scale, row))
        distinct_values = {round(value, 12) for value, _ in physical_values}
        if len(distinct_values) > 1:
            conflicts.append(f"Conflicting direct wavelength properties on {channel['name']}.")
            direct_complete = False
        elif physical_values:
            wavelength_nm = physical_values[0][0]
            partial_direct[channel["name"]] = wavelength_nm
            source = "; ".join(_source(row, source_file) for _, row in physical_values)
            direct_mapping.append(_mapping_row(
                channel, wavelength_nm, source,
                "explicit wavelength record; calibration traceability not independently assessed",
            ))
        else:
            direct_complete = False

    complete_mappings: list[tuple[str, list[dict[str, Any]]]] = []
    if direct_complete and len(direct_mapping) == len(ordered_channels):
        complete_mappings.append(("A", direct_mapping))

    if layout["layout_complete"] and ordered_channels:
        table_rows = [
            row for row in properties
            if row["scope"] in {"file", "group"}
            and _tokens(row["property_name"]) & physical_terms
            and _tokens(row["property_name"]) & {"lookup", "table"}
            and _unit_to_nm(row["unit"]) is not None
        ]
        for table_row in table_rows:
            values = _numeric_vector(table_row["raw_value"])
            if values is None or values.size != len(ordered_channels):
                continue
            order_rows = [
                row for row in properties
                if row["scope"] in {"file", "group"}
                and (row["scope"] == "file" or row["group_name"] == table_row["group_name"])
                and "order" in _tokens(row["property_name"])
                and (_tokens(row["property_name"]) & physical_terms
                     or _tokens(row["property_name"]) & {"lookup", "table"})
            ]
            order_values = [_lookup_order(row["raw_value"]) for row in order_rows]
            orders = set(order_values)
            if not order_rows or None in orders or len(orders) != 1:
                continue
            order = next(iter(orders))
            scale = _unit_to_nm(table_row["unit"])
            mapping = []
            order_source = "; ".join(
                _source(row, source_file) for row in order_rows if _lookup_order(row["raw_value"]) == order
            )
            for channel in ordered_channels:
                if order == "step_major":
                    table_index = channel["step"] * layout["sweeps_per_cycle"] + channel["sweep"]
                elif order == "sweep_major":
                    table_index = channel["sweep"] * layout["steps_per_sweep"] + channel["step"]
                else:
                    table_index = channel["footprint_index"]
                source = f"{_source(table_row, source_file)}; order={order_source}"
                mapping.append(_mapping_row(
                    channel, values[table_index] * scale, source,
                    "complete lookup table; calibration traceability not independently assessed",
                ))
            complete_mappings.append(("B", mapping))

    calibration_rows = [
        row for row in properties
        if row["scope"] in {"file", "group"}
        and "calibration" in _tokens(row["property_name"])
        and _tokens(row["property_name"]) & physical_terms
    ]
    starts = [row for row in calibration_rows if _tokens(row["property_name"]) & {"start", "origin"}]
    increments = [
        row for row in calibration_rows
        if "step" in _tokens(row["property_name"])
        and _tokens(row["property_name"]) & {"increment", "spacing"}
    ]
    offsets = [
        row for row in calibration_rows
        if "sweep" in _tokens(row["property_name"]) and "offset" in _tokens(row["property_name"])
    ]
    formulas = [
        row for row in calibration_rows
        if "formula" in _tokens(row["property_name"])
        and _parameter_formula_is_complete(row["raw_value"])
    ]
    if layout["layout_complete"] and ordered_channels and layout["sweeps_per_cycle"]:
        if len(starts) == len(increments) == len(offsets) == len(formulas) == 1:
            start, increment, offset, formula = starts[0], increments[0], offsets[0], formulas[0]
            start_value, increment_value, offset_value = (
                _finite_number(start), _finite_number(increment), _finite_number(offset)
            )
            scales = [_unit_to_nm(row["unit"]) for row in (start, increment, offset)]
            if None not in scales and None not in (start_value, increment_value, offset_value):
                start_nm, increment_nm, offset_nm = (
                    start_value * scales[0], increment_value * scales[1], offset_value * scales[2]
                )
                source = "; ".join(_source(row, source_file) for row in (start, increment, offset, formula))
                mapping = [
                    _mapping_row(
                        channel,
                        start_nm + channel["step"] * increment_nm + channel["sweep"] * offset_nm,
                        source,
                        "complete Step/Sweep formula; calibration traceability not independently assessed",
                    )
                    for channel in ordered_channels
                ]
                complete_mappings.append(("C", mapping))

    if partial_direct and len(partial_direct) < len(ordered_channels):
        for grade, mapping in complete_mappings:
            expected = {item["channel_name"]: item["wavelength_nm"] for item in mapping}
            for channel_name, value in partial_direct.items():
                if not np.isclose(value, expected[channel_name], rtol=0, atol=1e-9):
                    conflicts.append(f"Direct and grade {grade} wavelength evidence disagree on {channel_name}.")
    for i, (grade_a, mapping_a) in enumerate(complete_mappings):
        values_a = {item["channel_name"]: item["wavelength_nm"] for item in mapping_a}
        for grade_b, mapping_b in complete_mappings[i + 1:]:
            values_b = {item["channel_name"]: item["wavelength_nm"] for item in mapping_b}
            if any(not np.isclose(values_a[name], values_b[name], rtol=0, atol=1e-9) for name in values_a):
                conflicts.append(f"Complete grade {grade_a} and grade {grade_b} mappings disagree.")

    if conflicts:
        return {
            "grade": "D",
            "label": "Contradictory wavelength evidence",
            "evidence_class": "conflicting_calibration_candidates",
            "physical_wavelength_available": False,
            "mapping": [],
            "source_properties": candidates,
            "calibration_conflicts": sorted(set(conflicts)),
            "explanation": "Physical wavelength records or complete mappings disagree; no wavelength values are emitted.",
            "missing_calibration_information": [
                "Resolve the contradictory source values and document the selected calibration provenance."
            ],
        }

    if complete_mappings:
        rank = {"A": 0, "B": 1, "C": 2}
        grade, mapping = min(complete_mappings, key=lambda item: rank[item[0]])
        labels = {
            "A": ("Direct per-channel physical wavelength", "direct_physical_wavelength"),
            "B": ("Complete ordered physical wavelength lookup table", "complete_lookup_table"),
            "C": ("Explicit Step/Sweep parameterized wavelength calibration", "complete_parameterized_calibration"),
        }
        label, evidence_class = labels[grade]
        return {
            "grade": grade,
            "label": label,
            "evidence_class": evidence_class,
            "physical_wavelength_available": True,
            "mapping": mapping,
            "source_properties": candidates,
            "calibration_conflicts": [],
            "explanation": {
                "A": "Every channel has one consistent numeric physical-wavelength property with an explicit unit.",
                "B": "A complete unit-bearing table covers every channel and its explicit order defines Step/Sweep indexing.",
                "C": "Unit-bearing start, step increment, sweep offset, and a supported formula that repeats each ShiftCycle cover every channel.",
            }[grade],
            "missing_calibration_information": [],
        }

    indirect_terms = {"laser", "tuning", "frequency", "calibration", "grating", "steering", "voltage"}
    has_indirect = any(
        (_tokens(row["property_name"]) & indirect_terms)
        or (set(row.get("value_terms", [])) & indirect_terms)
        for row in candidates
    )
    has_unlabeled_array = any(
        "numeric_array" in row["candidate_reasons"]
        or "serialized_numeric_array" in row["candidate_reasons"]
        for row in candidates
    )
    scan_present = bool(scan_property_rows(properties)) or bool(ordered_channels)
    if has_unlabeled_array:
        evidence_class = "unlabeled_numeric_array_candidates"
        explanation = "Small numeric arrays were found, but their units, dimensions, and relation to Step/Sweep are not documented."
    elif has_indirect:
        evidence_class = "indirect_tuning_or_calibration_candidates"
        explanation = "Tuning/control or calibration candidates were found, but no complete physical mapping covers every Step/Sweep channel."
    elif scan_present:
        evidence_class = "scan_indices_only"
        explanation = "Step/Sweep indices and acquisition dimensions establish channel organization only; they do not establish physical wavelengths."
    else:
        evidence_class = "no_wavelength_evidence"
        explanation = "No physical wavelength, tuning, or calibration evidence was found in the inspected metadata."
    return {
        "grade": "D",
        "label": evidence_class.replace("_", " ").title(),
        "evidence_class": evidence_class,
        "physical_wavelength_available": False,
        "mapping": [],
        "source_properties": candidates,
        "calibration_conflicts": [],
        "explanation": explanation,
        "missing_calibration_information": [
            "A per-channel or per-pulse physical wavelength with an explicit unit, or a 256-entry lookup table with declared order.",
            "If using parameters: an explicit calibrated relationship that includes Step and Sweep, plus ShiftCycle applicability.",
            "Calibration traceability or a flight-specific calibration record to independently validate the wavelength scale.",
        ],
    }


def select_row_indices(total_rows: int, row_indices: list[int] | tuple[int, ...] | None = None) -> list[int]:
    if total_rows <= 0:
        raise ValueError("The TDMS group does not declare a positive row count.")
    if row_indices is None:
        values = sorted({0, total_rows // 2, total_rows - 1})
    else:
        values = [int(value) for value in row_indices]
    if not values:
        raise ValueError("Select at least one waveform row.")
    if len(set(values)) != len(values):
        raise ValueError(f"ROW_INDICES contains duplicates: {values}")
    invalid = [value for value in values if value < 0 or value >= total_rows]
    if invalid:
        raise IndexError(f"Waveform row indices out of range 0..{total_rows - 1}: {invalid}")
    return values


def read_channel_slice(channel: Any, start: int, stop: int) -> np.ndarray:
    """Read only a requested channel interval; never materialize the full TDMS channel."""
    if start < 0 or stop < start:
        raise ValueError(f"Invalid channel slice [{start}:{stop}]")
    values = np.asarray(channel[start:stop])
    if values.ndim != 1 or values.size != stop - start:
        raise ValueError(f"Channel returned {values.size} samples for requested slice [{start}:{stop}]")
    return values


def read_waveform_row(channel: Any, row_index: int, samples_per_row: int, total_rows: int) -> np.ndarray:
    if row_index < 0 or row_index >= total_rows:
        raise IndexError(f"row_index must be in 0..{total_rows - 1}, got {row_index}")
    if samples_per_row <= 0:
        raise ValueError("samples_per_row must be positive")
    start = row_index * samples_per_row
    return read_channel_slice(channel, start, start + samples_per_row)


def read_waveform_samples(
    tdms: Any, layout: dict[str, Any],
    row_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Read first, middle, and last rows for every channel using bounded slices."""
    if not layout["layout_complete"] or not layout["samples_per_row"] or not layout["rows_per_file"]:
        return {
            "status": "skipped_incomplete_layout",
            "selected_rows": [],
            "channel_statistics": [],
            "matrices": {},
        }
    selected_rows = select_row_indices(layout["rows_per_file"], row_indices)
    channels = sorted(layout["channels"], key=lambda item: item["footprint_index"])
    matrices = {}
    for row_index in selected_rows:
        matrix = np.empty(
            (layout["footprints_per_row"], layout["samples_per_row"]),
            dtype=np.dtype(channels[0]["dtype"]),
        )
        for channel in channels:
            samples = read_waveform_row(
                tdms[layout["group_name"]][channel["name"]],
                row_index,
                layout["samples_per_row"],
                layout["rows_per_file"],
            )
            matrix[channel["footprint_index"], :] = samples
        matrices[row_index] = matrix

    channel_statistics = []
    for channel in channels:
        values = np.concatenate([
            matrices[row][channel["footprint_index"]].reshape(-1) for row in selected_rows
        ])
        finite = np.isfinite(values)
        channel_statistics.append({
            "channel_name": channel["name"],
            "step": channel["step"],
            "sweep": channel["sweep"],
            "footprint_index": channel["footprint_index"],
            "sample_count": int(values.size),
            "finite_sample_count": int(finite.sum()),
            "finite_fraction": float(finite.mean()),
            "minimum_raw_count": float(np.min(values)),
            "maximum_raw_count": float(np.max(values)),
            "zero_fraction": float(np.mean(values == 0)),
        })
    all_values = np.concatenate([matrix.reshape(-1) for matrix in matrices.values()])
    finite = np.isfinite(all_values)
    return {
        "status": "sampled_rows_read",
        "selected_rows": selected_rows,
        "samples_per_row": layout["samples_per_row"],
        "sample_count_per_channel": layout["samples_per_row"] * len(selected_rows),
        "sampled_value_count": int(all_values.size),
        "sampled_bytes": int(all_values.nbytes),
        "finite_sample_count": int(finite.sum()),
        "finite_fraction": float(finite.mean()),
        "minimum_raw_count": float(np.min(all_values)),
        "maximum_raw_count": float(np.max(all_values)),
        "validity_note": "Finite numeric samples only; no invalid-sample sentinel is documented in the inspected metadata.",
        "channel_statistics": channel_statistics,
        "matrices": matrices,
    }


def audit_tdms_file(
    path: str | Path, row_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Audit one file's complete metadata/layout and bounded waveform samples."""
    path = Path(path)
    with open_tdms_metadata(path) as (tdms, reader_info):
        inventory = metadata_inventory(tdms)
        inventory["reader_status"].update(reader_info)
        layout = analyze_channel_layout(inventory)
        wavelength = wavelength_evidence(inventory, layout, source_file=path.name)
        try:
            waveform = read_waveform_samples(tdms, layout, row_indices)
        except Exception as exc:
            waveform = {
                "status": "sample_read_error",
                "selected_rows": [],
                "error": f"{type(exc).__name__}: {exc}",
                "channel_statistics": [],
                "matrices": {},
            }

    summary = {
        "file_name": path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "status": "audited",
        "metadata_source": inventory["reader_status"]["metadata_source"],
        "index_error": inventory["reader_status"]["index_error"],
        "groups": len(inventory["groups"]),
        "channels": sum(len(group["channels"]) for group in inventory["groups"]),
        "properties": len(inventory["property_rows"]),
        "channel_groups": [
            {
                "group": group["name"],
                "property_count": len(group["properties"]),
                "channel_count": len(group["channels"]),
                "channel_dtypes": sorted({channel["dtype"] for channel in group["channels"]}),
                "channel_lengths": sorted({channel["length"] for channel in group["channels"]}),
            }
            for group in inventory["groups"]
        ],
        "segment_count": inventory["reader_status"]["segment_count"],
        "segment_property_history": inventory["reader_status"]["segment_property_history"],
        "incomplete_final_segment": inventory["reader_status"]["incomplete_final_segment"],
        "channel_status_count": inventory["reader_status"]["channel_status_count"],
        "channel_status_mismatch_count": inventory["reader_status"]["channel_status_mismatch_count"],
        "channel_status_shortfall_samples": inventory["reader_status"]["channel_status_shortfall_samples"],
        "steps_per_sweep": layout["steps_per_sweep"],
        "sweeps_per_shift_cycle": layout["sweeps_per_cycle"],
        "rows_per_file": layout["rows_per_file"],
        "samples_per_row": layout["samples_per_row"],
        "layout_complete": layout["layout_complete"],
        "missing_step_sweep_count": len(layout["missing_step_sweep"]),
        "duplicate_step_sweep_count": len(layout["duplicate_step_sweep"]),
        "extra_step_sweep_count": len(layout["extra_step_sweep"]),
        "unrecognized_channel_count": len(layout["unrecognized_channels"]),
        "wavelength_grade": wavelength["grade"],
        "wavelength_evidence_class": wavelength["evidence_class"],
        "physical_wavelength_available": wavelength["physical_wavelength_available"],
        "wavelength_candidate_count": len(wavelength["source_properties"]),
        "waveform_sample_status": waveform["status"],
        "sampled_rows": waveform["selected_rows"],
        "sampled_value_count": waveform.get("sampled_value_count", 0),
        "finite_fraction": waveform.get("finite_fraction"),
        "minimum_raw_count": waveform.get("minimum_raw_count"),
        "maximum_raw_count": waveform.get("maximum_raw_count"),
        "error": waveform.get("error"),
    }
    return {
        "summary": summary,
        "inventory": inventory,
        "layout": layout,
        "wavelength_evidence": wavelength,
        "waveform_samples": waveform,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )


def audit_all_tdms(
    data_dir: str | Path, output_dir: str | Path,
    row_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Audit every TDMS file, preserve per-file errors, and write a comparison table."""
    files = discover_tdms_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .tdms files were found below {Path(data_dir).resolve()}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty audit output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    columns = [
        "file_name", "size_bytes", "status", "metadata_source", "segment_count",
        "channel_status_count", "channel_status_mismatch_count", "channel_status_shortfall_samples",
        "groups", "channels", "steps_per_sweep", "sweeps_per_shift_cycle",
        "rows_per_file", "samples_per_row", "layout_complete",
        "wavelength_grade", "wavelength_evidence_class", "physical_wavelength_available",
        "waveform_sample_status", "sampled_rows", "sampled_value_count",
        "finite_fraction", "minimum_raw_count", "maximum_raw_count", "error",
    ]
    for path in files:
        file_dir = output_dir / path.stem
        file_dir.mkdir()
        try:
            result = audit_tdms_file(path, row_indices)
            summary = result["summary"]
            waveform = {key: value for key, value in result["waveform_samples"].items() if key != "matrices"}
            _write_json(file_dir / "metadata_inventory.json", result["inventory"])
            _write_json(file_dir / "channel_layout.json", result["layout"])
            _write_json(file_dir / "wavelength_evidence.json", result["wavelength_evidence"])
            _write_json(file_dir / "waveform_sample_summary.json", waveform)
            with (file_dir / "waveform_sample_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
                rows = result["waveform_samples"]["channel_statistics"]
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
                    "channel_name", "step", "sweep", "footprint_index", "sample_count",
                    "finite_sample_count", "finite_fraction", "minimum_raw_count",
                    "maximum_raw_count", "zero_fraction",
                ])
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            summary = {
                "file_name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "status": "audit_error",
                "metadata_source": None,
                "segment_count": None,
                "channel_status_count": None,
                "channel_status_mismatch_count": None,
                "channel_status_shortfall_samples": None,
                "groups": None,
                "channels": None,
                "steps_per_sweep": None,
                "sweeps_per_shift_cycle": None,
                "rows_per_file": None,
                "samples_per_row": None,
                "layout_complete": False,
                "wavelength_grade": None,
                "wavelength_evidence_class": None,
                "physical_wavelength_available": False,
                "waveform_sample_status": "not_attempted",
                "sampled_rows": [],
                "sampled_value_count": 0,
                "finite_fraction": None,
                "minimum_raw_count": None,
                "maximum_raw_count": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_json(file_dir / "audit_summary.json", summary)
        summaries.append(summary)

    with (output_dir / "cross_file_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in summaries)
    result = {
        "data_dir": str(Path(data_dir).resolve()),
        "file_count": len(files),
        "audited_count": sum(row["status"] == "audited" for row in summaries),
        "error_count": sum(row["status"] != "audited" for row in summaries),
        "summaries": summaries,
    }
    _write_json(output_dir / "batch_audit.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit all CASALS L0 TDMS files in one directory.")
    parser.add_argument("--all", action="store_true", help="audit every .tdms file under --data-dir")
    parser.add_argument("--data-dir", default="data/raw/tdms")
    parser.add_argument("--output-dir", required=True, help="new or empty directory for audit artifacts")
    parser.add_argument("--rows", help="comma-separated row indices; default is first, middle, and last")
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("pass --all to audit every discovered TDMS file")
    rows = [int(value.strip()) for value in args.rows.split(",") if value.strip()] if args.rows else None
    result = audit_all_tdms(args.data_dir, args.output_dir, rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

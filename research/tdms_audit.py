"""Small, read-only helpers for auditing CASALS L0 TDMS files."""

from __future__ import annotations

import base64
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


def metadata_inventory(tdms: Any) -> dict[str, Any]:
    """Capture every file, group, and channel property plus channel shape metadata."""
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
    return {
        "file_properties": _jsonable(file_properties),
        "groups": groups,
        "property_rows": property_rows,
    }


def scan_property_rows(property_rows: list[dict[str, Any]], *, wavelength_only: bool = False) -> list[dict[str, Any]]:
    """Select metadata fields by normalized keyword tokens, retaining the full inventory separately."""
    terms = _WAVELENGTH_TERMS if wavelength_only else _SCAN_TERMS
    return [row for row in property_rows if _tokens(row["property_name"]) & terms]


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
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def analyze_channel_layout(inventory: dict[str, Any], group_name: str | None = None) -> dict[str, Any]:
    """Parse Sweep{s}Step{st}, compare with declared dimensions, and derive viewer order."""
    groups = inventory["groups"]
    if not groups:
        raise ValueError("TDMS file has no groups.")
    group = next((item for item in groups if item["name"] == group_name), None) if group_name else None
    group = group or next((item for item in groups if item["name"] == "CASALS"), groups[0])
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
    if unit in {"nm"}:
        return 1.0
    if unit in {"um", "µm", "μm"}:
        return 1000.0
    if unit == "angstrom":
        return 0.1
    return None


def wavelength_evidence(inventory: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Conservatively classify direct wavelength evidence or a named calibration mapping."""
    properties = inventory["property_rows"]
    candidates = scan_property_rows(properties, wavelength_only=True)
    physical_name_terms = {"wvl", "wavelength", "lambda"}
    channel_map: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        if row["scope"] == "channel" and row["channel_name"]:
            channel_map.setdefault(row["channel_name"], []).append(row)

    ordered_channels = sorted(
        (item for item in layout["channels"] if item["footprint_index"] is not None),
        key=lambda item: item["footprint_index"],
    )
    direct = []
    if layout["layout_complete"] and ordered_channels:
        for channel in ordered_channels:
            candidates_for_channel = channel_map.get(channel["name"], [])
            selected = next((
                row for row in candidates_for_channel
                if _unit_to_nm(row["unit"]) is not None
                and (_tokens(row["property_name"]) & physical_name_terms)
                and isinstance(row["raw_value"], (int, float))
                and not isinstance(row["raw_value"], bool)
                and np.isfinite(row["raw_value"])
                and not (_tokens(row["property_name"]) & {"steps", "step"})
            ), None)
            if selected is None:
                direct = []
                break
            direct.append({
                "footprint_index": channel["footprint_index"],
                "channel_name": channel["name"],
                "sweep": channel["sweep"],
                "step": channel["step"],
                "wavelength_nm": float(selected["raw_value"]) * _unit_to_nm(selected["unit"]),
                "source_property": selected["property_name"],
                "source_scope": selected["scope"],
                "source_unit": selected["unit"],
            })
        if len(direct) == len(ordered_channels):
            return {
                "grade": "A",
                "label": "Explicit physical wavelength",
                "physical_wavelength_available": True,
                "mapping": direct,
                "source_properties": candidates,
                "explanation": "Every parsed channel has an explicit numeric wavelength property with a physical unit.",
                "missing_calibration_information": [],
            }

    if layout["layout_complete"] and ordered_channels:
        for row in candidates:
            if row["scope"] not in {"file", "group"} or _unit_to_nm(row["unit"]) is None:
                continue
            tokens = _tokens(row["property_name"])
            if not tokens & physical_name_terms or not tokens & {"lookup", "table", "channel", "track", "footprint", "beam", "step", "steps"}:
                continue
            try:
                values = np.asarray(row["raw_value"], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite(values)):
                continue
            if values.size == layout["footprints_per_row"] and tokens & {"lookup", "table", "channel", "track", "footprint", "beam"}:
                mapped_values = {channel["footprint_index"]: values[channel["footprint_index"]] for channel in ordered_channels}
            elif values.size == layout["steps_per_sweep"] and tokens & {"step", "steps"}:
                mapped_values = {channel["footprint_index"]: values[channel["step"]] for channel in ordered_channels}
            else:
                continue
            scale = _unit_to_nm(row["unit"])
            mapping = [{
                "footprint_index": channel["footprint_index"],
                "channel_name": channel["name"],
                "sweep": channel["sweep"],
                "step": channel["step"],
                "wavelength_nm": float(mapped_values[channel["footprint_index"]]) * scale,
                "source_property": row["property_name"],
                "source_scope": row["scope"],
                "source_unit": row["unit"],
            } for channel in ordered_channels]
            return {
                "grade": "A",
                "label": "Explicit physical wavelength",
                "physical_wavelength_available": True,
                "mapping": mapping,
                "source_properties": candidates,
                "explanation": "An explicitly named, unit-bearing wavelength lookup table maps each channel or wavelength step.",
                "missing_calibration_information": [],
            }

    # B is limited to explicitly named calibration start/increment parameters with physical units.
    calibration_rows = [row for row in candidates if row["scope"] in {"file", "group"}
                        and "calibration" in _tokens(row["property_name"])
                        and _tokens(row["property_name"]) & physical_name_terms]
    start_row = next((row for row in calibration_rows
                      if _tokens(row["property_name"]) & {"start", "origin"}
                      and _unit_to_nm(row["unit"]) is not None
                      and isinstance(row["raw_value"], (int, float))), None)
    increment_row = next((row for row in calibration_rows
                          if _tokens(row["property_name"]) & {"increment", "interval", "spacing"}
                          and _unit_to_nm(row["unit"]) is not None
                          and isinstance(row["raw_value"], (int, float))), None)
    if (layout["layout_complete"] and ordered_channels and start_row and increment_row
            and start_row["unit"] == increment_row["unit"]):
        scale = _unit_to_nm(start_row["unit"])
        mapping = [{
            "footprint_index": channel["footprint_index"],
            "channel_name": channel["name"],
            "sweep": channel["sweep"],
            "step": channel["step"],
            "wavelength_nm": (float(start_row["raw_value"])
                              + float(increment_row["raw_value"]) * channel["step"]) * scale,
            "source_property": f"{start_row['property_name']} + step * {increment_row['property_name']}",
            "source_scope": ";".join(sorted({start_row["scope"], increment_row["scope"]})),
            "source_unit": start_row["unit"],
        } for channel in ordered_channels]
        return {
            "grade": "B",
            "label": "Calibrated wavelength mapping",
            "physical_wavelength_available": True,
            "mapping": mapping,
            "source_properties": candidates,
            "explanation": "Explicitly named wavelength calibration start and increment properties map each step in a physical unit.",
            "missing_calibration_information": [],
        }

    scan_present = bool(scan_property_rows(properties)) or bool(ordered_channels)
    grade = "C" if scan_present else "D"
    missing = [
        "A per-channel or per-step physical wavelength value with units",
        "A wavelength calibration lookup or calibrated start and increment in physical units",
        "A tuning-voltage/frequency or beam-steering calibration tied to the channel index",
    ]
    return {
        "grade": grade,
        "label": "Scan-step information only" if grade == "C" else "No usable wavelength evidence",
        "physical_wavelength_available": False,
        "mapping": [],
        "source_properties": candidates,
        "explanation": (
            "Scan/channel indices are present, but the inspected properties do not provide a complete physical wavelength mapping."
            if grade == "C" else "No physical wavelength or usable wavelength scan metadata was found."
        ),
        "missing_calibration_information": missing,
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

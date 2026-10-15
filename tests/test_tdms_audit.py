from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

from research.tdms_audit import (
    _unit,
    analyze_channel_layout,
    audit_all_tdms,
    audit_raw_data_samples,
    channel_property_variation,
    discover_tdms_files,
    metadata_inventory,
    metadata_value_candidates,
    open_tdms_metadata,
    read_waveform_row,
    read_waveform_samples,
    resolve_tdms_reference,
    select_row_indices,
    select_tdms_file,
    wavelength_evidence,
)


def _row(scope: str, name: str, value, group: str | None = None, channel: str | None = None):
    raw = value.tolist() if isinstance(value, np.ndarray) else value
    return {
        "scope": scope,
        "property_name": name,
        "raw_data_type": type(value).__name__,
        "raw_value": raw,
        "readable_value": repr(raw),
        "unit": _unit(name),
        "group_name": group,
        "channel_name": channel,
    }


def _inventory(
    *,
    steps: int = 3,
    sweeps: int = 2,
    rows: int = 3,
    samples: int = 4,
    group_properties: dict | None = None,
    channel_properties: dict | None = None,
    channel_names: list[str] | None = None,
    groups_before: bool = False,
):
    props = {
        "# WVLSteps/Sweep": steps,
        "# Sweeps/ShiftCycle": sweeps,
        "Sweeps/File": rows,
    }
    props.update(group_properties or {})
    names = channel_names or [
        f"Sweep{sweep}Step{step}"
        for sweep in range(sweeps)
        for step in range(steps)
    ]
    casals_channels = []
    property_rows = [_row("file", "Run label", "synthetic fixture")]
    for index, name in enumerate(names):
        match = __import__("re").fullmatch(r"Sweep(\d+)Step(\d+)", name, __import__("re").I)
        channel_props = (channel_properties or {}).get(name, {})
        channel = {
            "name": name,
            "length": rows * samples + (1 if index == len(names) - 1 and name == "unequal" else 0),
            "data_type": "Int16",
            "dtype": "int16",
            "properties": dict(channel_props),
        }
        casals_channels.append(channel)
        property_rows.extend(
            _row("channel", key, value, "CASALS", name)
            for key, value in channel_props.items()
        )
    group = {"name": "CASALS", "properties": props, "channels": casals_channels}
    property_rows.extend(_row("group", key, value, "CASALS") for key, value in props.items())
    groups = [group]
    if groups_before:
        other_props = {"Description": "secondary group"}
        other = {
            "name": "OTHER",
            "properties": other_props,
            "channels": [{"name": "Aux", "length": 1, "data_type": "Int16", "dtype": "int16", "properties": {}}],
        }
        groups = [other, group]
        property_rows.extend(_row("group", key, value, "OTHER") for key, value in other_props.items())
    return {
        "file_properties": {"Run label": "synthetic fixture"},
        "groups": groups,
        "property_rows": property_rows,
    }


def _write_tdms(
    path: Path,
    *,
    steps: int = 3,
    sweeps: int = 2,
    rows: int = 3,
    samples: int = 4,
    group_properties: dict | None = None,
    channel_properties: dict | None = None,
    missing: tuple[int, int] | None = None,
    unequal_lengths: bool = False,
    add_other_group: bool = False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    props = {
        "# WVLSteps/Sweep": steps,
        "# Sweeps/ShiftCycle": sweeps,
        "Sweeps/File": rows,
    }
    props.update(group_properties or {})
    objects = [RootObject(properties={"Run label": "small test fixture"})]
    if add_other_group:
        objects.extend([
            GroupObject("OTHER", properties={"Description": "secondary group"}),
            ChannelObject("OTHER", "Aux", np.arange(4, dtype=np.int16)),
        ])
    objects.append(GroupObject("CASALS", properties=props))
    for sweep in range(sweeps):
        for step in range(steps):
            if missing == (sweep, step):
                continue
            name = f"Sweep{sweep}Step{step}"
            length = rows * samples + (1 if unequal_lengths and (sweep, step) == (sweeps - 1, steps - 1) else 0)
            values = np.arange(length, dtype=np.int16) + 100 * sweep + 10 * step
            objects.append(ChannelObject(
                "CASALS", name, values, properties=(channel_properties or {}).get(name, {})
            ))
    with TdmsWriter(path) as writer:
        writer.write_segment(objects)
    return path


def _write_partial_tail_tdms(path: Path) -> Path:
    """Make a tiny fixed-width file with three full raw chunks and a 2-sample tail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TdmsWriter(path) as writer:
        writer.write_segment([
            RootObject(properties={"Run label": "partial-tail fixture"}),
            GroupObject("CASALS", properties={
                "# WVLSteps/Sweep": 1, "# Sweeps/ShiftCycle": 2, "Sweeps/File": 5,
            }),
            ChannelObject("CASALS", "Sweep0Step0", np.arange(4, dtype=np.int16)),
            ChannelObject("CASALS", "Sweep1Step0", np.arange(4, dtype=np.int16) + 10),
        ])
    payload = bytearray(path.read_bytes())
    raw_start = struct.unpack_from("<Q", payload, 20)[0] + 28
    channels = 2
    samples_per_chunk = 4
    raw_chunks = []
    for chunk in range(3):
        for channel in range(channels):
            values = np.arange(samples_per_chunk, dtype=np.int16) + 100 * chunk + 10 * channel
            raw_chunks.append(values.tobytes())
    for channel in range(channels):
        raw_chunks.append((np.arange(2, dtype=np.int16) + 300 + 10 * channel).tobytes())
    raw_payload = b"".join(raw_chunks)
    payload = payload[:raw_start] + raw_payload
    struct.pack_into("<Q", payload, 12, len(payload) - 28)
    path.write_bytes(payload)
    index = bytearray(payload[:raw_start])
    index[:4] = b"TDSh"
    Path(f"{path}_index").write_bytes(index)
    return path


def test_inventory_preserves_properties_and_reports_segment_limitations(tmp_path: Path):
    path = tmp_path / "two_segments.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([
            RootObject(properties={"Run label": "first"}),
            GroupObject("CASALS", properties={"Revision": "first", "# WVLSteps/Sweep": 1,
                                               "# Sweeps/ShiftCycle": 1, "Sweeps/File": 2}),
            ChannelObject("CASALS", "Sweep0Step0", np.arange(4, dtype=np.int16),
                          properties={"gain": 1.5, "enabled": True}),
        ])
        writer.write_segment([
            GroupObject("CASALS", properties={"Revision": "second"}),
            ChannelObject("CASALS", "Sweep0Step0", np.arange(4, dtype=np.int16) + 4),
        ])

    with open_tdms_metadata(path) as (tdms, reader_info):
        inventory = metadata_inventory(tdms)
    assert reader_info == {"metadata_source": "tdms", "index_error": None}
    assert inventory["reader_status"]["segment_count"] == 2
    assert inventory["reader_status"]["segment_property_history"] == "merged_final_properties_only"
    assert inventory["reader_status"]["incomplete_final_segment"] is False
    assert inventory["groups"][0]["properties"]["Revision"] == "second"
    assert inventory["groups"][0]["channels"][0]["length"] == 8
    rows = {(row["scope"], row["property_name"]): row for row in inventory["property_rows"]}
    assert rows[("file", "Run label")]["raw_value"] == "first"
    assert rows[("channel", "gain")]["raw_value"] == 1.5
    assert rows[("channel", "enabled")]["raw_data_type"] == "bool"


def test_inventory_preserves_per_channel_segment_read_status():
    class Channel:
        name = "Sweep0Step0"
        properties = {}
        data_type = "Int16"
        dtype = np.dtype("int16")

        def __len__(self):
            return 8

    class Group:
        name = "CASALS"
        properties = {"# WVLSteps/Sweep": 1, "# Sweeps/ShiftCycle": 1, "Sweeps/File": 2}

        def channels(self):
            return [Channel()]

    class Reader:
        _segments = [object()]

    class File:
        properties = {"name": "fixture"}
        _reader = Reader()
        file_status = type("Status", (), {
            "incomplete_final_segment": False,
            "channel_statuses": {
                "/'CASALS'/'Sweep0Step0'": type("ChannelStatus", (), {
                    "expected_length": 2816, "read_length": 2560,
                })(),
            },
        })()

        def groups(self):
            return [Group()]

    status = metadata_inventory(File())["reader_status"]
    assert status["channel_status_count"] == 1
    assert status["channel_status_mismatch_count"] == 1
    assert status["channel_status_shortfall_samples"] == 256
    assert status["channel_status_records"] == [{
        "channel_path": "/'CASALS'/'Sweep0Step0'",
        "expected_length": 2816,
        "read_length": 2560,
        "shortfall_samples": 256,
    }]


def test_bad_index_falls_back_to_raw_tdms_and_keeps_slice_reads_available(tmp_path: Path):
    path = _write_tdms(tmp_path / "fallback.tdms")
    Path(f"{path}_index").write_bytes(b"invalid index")
    with open_tdms_metadata(path) as (tdms, info):
        assert info["metadata_source"] == "tdms_raw_fallback"
        assert info["index_error"]
        values = np.asarray(tdms["CASALS"]["Sweep0Step0"][:4])
    assert values.tolist() == [0, 1, 2, 3]


def test_layout_validates_step_sweep_counts_lengths_and_file_order():
    inventory = _inventory()
    layout = analyze_channel_layout(inventory)
    assert layout["layout_complete"]
    assert layout["steps_per_sweep"] == 3
    assert layout["sweeps_per_cycle"] == 2
    assert layout["footprints_per_row"] == 6
    assert layout["channel_order_in_file"] == "sweep-major then step"
    assert layout["file_order_matches_step_major_then_sweep"] is False
    index_by_name = {channel["name"]: channel["footprint_index"] for channel in layout["channels"]}
    assert index_by_name["Sweep0Step0"] == 0
    assert index_by_name["Sweep1Step0"] == 1
    assert index_by_name["Sweep0Step1"] == 2
    assert index_by_name["Sweep1Step2"] == 5
    assert layout["samples_per_row"] == 4
    assert layout["channel_structure_complete"] is True
    assert layout["row_layout_valid"] is True

    missing = _inventory(channel_names=[
        "Sweep0Step0", "Sweep0Step1", "Sweep0Step2",
        "Sweep1Step0", "Sweep1Step1",
    ])
    assert analyze_channel_layout(missing)["missing_step_sweep"] == [[1, 2]]

    unequal = _inventory(channel_names=[
        "Sweep0Step0", "Sweep0Step1", "Sweep0Step2",
        "Sweep1Step0", "Sweep1Step1", "unequal",
    ])
    unequal_layout = analyze_channel_layout(unequal)
    assert not unequal_layout["channel_lengths_consistent"]
    assert not unequal_layout["rows_file_divide_channel_length"]
    assert not unequal_layout["layout_complete"]

    unresolved_inventory = _inventory()
    unresolved_inventory["groups"][0]["properties"]["Sweeps/File"] = 5
    structurally_complete_but_unresolved_rows = analyze_channel_layout(unresolved_inventory)
    assert structurally_complete_but_unresolved_rows["channel_structure_complete"] is True
    assert structurally_complete_but_unresolved_rows["row_layout_valid"] is False
    assert structurally_complete_but_unresolved_rows["layout_complete"] is False


def test_layout_handles_multiple_groups_and_reports_extra_duplicate_or_unknown_channels():
    inventory = _inventory(groups_before=True)
    layout = analyze_channel_layout(inventory)
    assert layout["group_name"] == "CASALS"
    with pytest.raises(ValueError, match="does not exist"):
        analyze_channel_layout(inventory, "missing")

    names = [
        "Sweep0Step0", "Sweep0Step1", "Sweep0Step2",
        "Sweep1Step0", "Sweep1Step1", "Sweep1Step2",
        "Sweep0Step0", "Sweep9Step9", "not_a_scan_channel",
    ]
    broken = analyze_channel_layout(_inventory(channel_names=names))
    assert broken["duplicate_channel_names"] == ["Sweep0Step0"]
    assert broken["duplicate_step_sweep"] == [[0, 0]]
    assert broken["extra_step_sweep"] == [[9, 9]]
    assert broken["unrecognized_channels"] == ["not_a_scan_channel"]
    assert not broken["layout_complete"]


def test_direct_channel_wavelengths_require_units_and_include_provenance():
    properties = {
        f"Sweep{sweep}Step{step}": {"Wavelength (nm)": 1040.0 + step * 0.1 + sweep * 0.01}
        for sweep in range(2) for step in range(3)
    }
    inventory = _inventory(channel_properties=properties)
    layout = analyze_channel_layout(inventory)
    evidence = wavelength_evidence(inventory, layout, source_file="run.tdms")
    assert evidence["grade"] == "A"
    assert evidence["physical_wavelength_available"]
    assert len(evidence["mapping"]) == 6
    mapped = next(row for row in evidence["mapping"] if row["channel_name"] == "Sweep1Step2")
    assert mapped["wavelength_nm"] == pytest.approx(1040.21)
    assert "run.tdms" in mapped["source"]
    assert "scope=channel" in mapped["source"]
    assert "channel=Sweep1Step2" in mapped["source"]
    assert "property=Wavelength (nm)" in mapped["source"]
    assert mapped["calibration_status"].startswith("explicit wavelength record")

    micro = _inventory(channel_properties={
        f"Sweep{sweep}Step{step}": {"Wavelength (\u00b5m)": 1.04}
        for sweep in range(2) for step in range(3)
    })
    micro_evidence = wavelength_evidence(micro, analyze_channel_layout(micro))
    assert micro_evidence["mapping"][0]["wavelength_nm"] == pytest.approx(1040.0)

    no_unit = _inventory(channel_properties={
        f"Sweep{sweep}Step{step}": {"Wavelength": 1040.0}
        for sweep in range(2) for step in range(3)
    })
    missing_unit = wavelength_evidence(no_unit, analyze_channel_layout(no_unit))
    assert not missing_unit["physical_wavelength_available"]
    assert missing_unit["mapping"] == []


def test_complete_256_entry_lookup_requires_explicit_table_order():
    steps, sweeps = 32, 8
    values = [1000.0 + index * 0.01 for index in range(steps * sweeps)]
    inventory = _inventory(
        steps=steps,
        sweeps=sweeps,
        rows=1,
        group_properties={
            "Wavelength lookup table (nm)": values,
            "Wavelength lookup order": "step-major, sweep-minor",
        },
    )
    layout = analyze_channel_layout(inventory)
    evidence = wavelength_evidence(inventory, layout, source_file="lookup.tdms")
    assert evidence["grade"] == "B"
    assert len(evidence["mapping"]) == 256
    mapped = next(row for row in evidence["mapping"] if row["channel_name"] == "Sweep7Step31")
    assert mapped["footprint_index"] == 255
    assert mapped["wavelength_nm"] == pytest.approx(values[255])
    assert "Wavelength lookup table (nm)" in mapped["source"]
    assert "Wavelength lookup order" in mapped["source"]
    assert mapped["calibration_status"].startswith("complete lookup table")

    incomplete = _inventory(
        steps=steps,
        sweeps=sweeps,
        rows=1,
        group_properties={"Wavelength step table (nm)": list(range(steps))},
    )
    incomplete_evidence = wavelength_evidence(incomplete, analyze_channel_layout(incomplete))
    assert not incomplete_evidence["physical_wavelength_available"]
    assert incomplete_evidence["mapping"] == []


def test_parameterized_calibration_requires_explicit_step_sweep_formula_and_cycle_semantics():
    props = {
        "Laser wavelength calibration start (nm)": 1040.0,
        "Laser wavelength calibration step increment (nm)": 0.1,
        "Laser wavelength calibration sweep offset (nm)": 1.0,
        "Laser wavelength calibration formula": (
            "start + step * increment + sweep * sweep_offset; repeats each shift cycle"
        ),
    }
    inventory = _inventory(steps=32, sweeps=8, rows=1, group_properties=props)
    evidence = wavelength_evidence(inventory, analyze_channel_layout(inventory))
    assert evidence["grade"] == "C"
    assert evidence["physical_wavelength_available"]
    assert len(evidence["mapping"]) == 256
    mapped = next(row for row in evidence["mapping"] if row["channel_name"] == "Sweep7Step31")
    assert mapped["wavelength_nm"] == pytest.approx(1050.1)
    assert "sweep offset" in mapped["source"].lower()
    assert "formula" in mapped["source"].lower()

    step_only = _inventory(steps=32, sweeps=8, rows=1, group_properties={
        "Laser wavelength calibration start (nm)": 1040.0,
        "Laser wavelength calibration step increment (nm)": 0.1,
    })
    step_only_evidence = wavelength_evidence(step_only, analyze_channel_layout(step_only))
    assert not step_only_evidence["physical_wavelength_available"]
    assert step_only_evidence["mapping"] == []


def test_control_values_serialized_configs_and_unlabeled_arrays_remain_candidates_only():
    config = '{"laser_tuning_voltage": 2.5, "wavelength_nm": 1040}'
    inventory = _inventory(group_properties={
        "Instrument configuration": config,
        "Unknown vector": [1.0, 2.0, 3.0],
        "Laser tuning voltage (V)": 2.5,
    })
    candidates = metadata_value_candidates(inventory["property_rows"])
    config_row = next(row for row in candidates if row["property_name"] == "Instrument configuration")
    assert "string_value" in config_row["candidate_reasons"]
    assert {"laser", "tuning", "wavelength"} <= set(config_row["value_terms"])
    vector_row = next(row for row in candidates if row["property_name"] == "Unknown vector")
    assert "numeric_array" in vector_row["candidate_reasons"]
    assert vector_row["numeric_value_count"] == 3

    evidence = wavelength_evidence(inventory, analyze_channel_layout(inventory))
    assert evidence["grade"] == "D"
    assert not evidence["physical_wavelength_available"]
    assert evidence["mapping"] == []
    assert evidence["evidence_class"] in {
        "unlabeled_numeric_array_candidates", "indirect_tuning_or_calibration_candidates"
    }

    dac_inventory = _inventory(group_properties={"DAC": 0.25, "laser DAC code": 125.0})
    dac_evidence = wavelength_evidence(dac_inventory, analyze_channel_layout(dac_inventory))
    assert dac_evidence["grade"] == "D"
    assert dac_evidence["mapping"] == []


def test_contradictory_physical_wavelength_sources_never_emit_mapping():
    direct = {
        f"Sweep{sweep}Step{step}": {"Wavelength (nm)": 1040.0 + step * 0.1 + sweep}
        for sweep in range(2) for step in range(3)
    }
    table = [1040.0 + index * 0.1 for index in range(6)]
    inventory = _inventory(
        group_properties={
            "Wavelength lookup table (nm)": table,
            "Wavelength lookup order": "step-major, sweep-minor",
        },
        channel_properties=direct,
    )
    evidence = wavelength_evidence(inventory, analyze_channel_layout(inventory))
    assert evidence["grade"] == "D"
    assert not evidence["physical_wavelength_available"]
    assert evidence["mapping"] == []
    assert evidence["calibration_conflicts"]


def test_duplicate_direct_values_are_reported_as_conflicting():
    properties = {
        f"Sweep{sweep}Step{step}": {
            "Wavelength (nm)": 1040.0 + step * 0.1,
            "Lambda (nm)": 1041.0 + step * 0.1,
        }
        for sweep in range(2) for step in range(3)
    }
    inventory = _inventory(channel_properties=properties)
    evidence = wavelength_evidence(inventory, analyze_channel_layout(inventory))
    assert evidence["grade"] == "D"
    assert evidence["mapping"] == []
    assert evidence["calibration_conflicts"]


def test_indices_only_are_not_wavelengths_and_channel_property_variation_is_retained():
    inventory = _inventory(channel_properties={
        f"Sweep{sweep}Step{step}": {"gain": 1 if sweep == 0 else 2}
        for sweep in range(2) for step in range(3)
    })
    layout = analyze_channel_layout(inventory)
    evidence = wavelength_evidence(inventory, layout)
    assert evidence["grade"] == "D"
    assert evidence["evidence_class"] == "scan_indices_only"
    assert evidence["mapping"] == []
    assert "wavelength_nm" not in json.dumps(evidence["mapping"])
    variation = channel_property_variation(inventory)
    gain = next(row for row in variation if row["property_name"] == "gain")
    assert gain["channel_count"] == 6
    assert gain["distinct_value_count"] == 2


class TrackingChannel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int16)
        self.requested = []

    def __getitem__(self, key):
        self.requested.append(key)
        return self.values[key]


def test_waveform_reads_use_bounded_slices_for_every_channel():
    inventory = _inventory(rows=3, samples=4)
    layout = analyze_channel_layout(inventory)
    group = {
        channel["name"]: TrackingChannel(np.arange(12, dtype=np.int16) + index)
        for index, channel in enumerate(layout["channels"])
    }
    result = read_waveform_samples({"CASALS": group}, layout)
    assert result["selected_rows"] == [0, 1, 2]
    assert result["sampled_value_count"] == 6 * 3 * 4
    assert result["finite_fraction"] == 1.0
    assert len(result["channel_statistics"]) == 6
    for channel in group.values():
        assert channel.requested == [slice(0, 4), slice(4, 8), slice(8, 12)]

    one = TrackingChannel(np.arange(12, dtype=np.int16))
    values = read_waveform_row(one, row_index=2, samples_per_row=4, total_rows=3)
    assert values.tolist() == [8, 9, 10, 11]
    assert one.requested == [slice(8, 12)]


def test_file_discovery_and_selection_exclude_index_sidecars(tmp_path: Path):
    data_dir = tmp_path / "data"
    first = _write_tdms(data_dir / "a.tdms")
    second = _write_tdms(data_dir / "b.tdms")
    Path(f"{first}_index").write_bytes(b"index")
    assert discover_tdms_files(data_dir) == [first, second]
    with pytest.raises(ValueError, match="choose one explicitly"):
        select_tdms_file(tmp_path, data_dir, None)
    assert select_tdms_file(tmp_path, data_dir, "data/a.tdms") == first.resolve()
    with pytest.raises(ValueError, match=r"Expected a \.tdms file"):
        select_tdms_file(tmp_path, data_dir, "a.tdms_index")
    assert select_tdms_file(tmp_path, data_dir, "data/b.tdms") == second.resolve()


def test_tdms_reference_requires_unique_exact_basename_match(tmp_path: Path):
    first = tmp_path / "a" / "source.tdms"
    second = tmp_path / "b" / "other.tdms"
    assert resolve_tdms_reference(r"folder\source.tdms", [first, second]) == first
    assert resolve_tdms_reference("source.tdms.bak", [first, second]) is None
    assert resolve_tdms_reference("missing.tdms", [first, second]) is None
    assert resolve_tdms_reference("source.tdms", [first, tmp_path / "duplicate" / "source.tdms"]) is None


def test_raw_audit_verifies_selected_full_chunks_and_partial_tail_without_claiming_rows(tmp_path: Path):
    path = _write_partial_tail_tdms(tmp_path / "partial_tail.tdms")
    with open_tdms_metadata(path) as (tdms, reader_info):
        inventory = metadata_inventory(tdms)
        inventory["reader_status"].update(reader_info)
        layout = analyze_channel_layout(inventory)
        raw = audit_raw_data_samples(tdms, path, layout)

    assert raw["lead_in_matches_reader_and_file"] is True
    assert raw["index_sidecar"]["status"] == "matches_main_metadata_prefix"
    assert raw["status"] == "partial_final_chunk_verified"
    assert raw["full_chunk_count"] == 3
    assert raw["candidate_complete_storage_chunks"] == 3
    assert raw["partial_final_chunk_samples"] == 2
    assert raw["channel_lengths_match_raw_byte_count"] is True
    assert raw["sampled_chunk_indices"] == [0, 1, 2]
    assert raw["sampled_value_count_checked"] == 28
    assert raw["valid_prefix_rows"] is None
    assert raw["row_layout_valid"] is False
    assert raw["all_selected_slices_match_raw_bytes_and_raw_only_reader"] is True
    assert np.array_equal(raw["partial_final_chunk_matrix"], np.array([[300, 301], [310, 311]]))


def test_batch_counts_raw_integrity_failure_separately_from_metadata_success(tmp_path: Path, monkeypatch):
    import research.tdms_audit as tdms_audit

    data_dir = tmp_path / "tdms"
    _write_tdms(data_dir / "raw_failure.tdms")

    def fail_raw_read(*args, **kwargs):
        return {"status": "read_integrity_failed", "error": "synthetic read mismatch"}

    monkeypatch.setattr(tdms_audit, "audit_raw_data_samples", fail_raw_read)
    result = audit_all_tdms(data_dir, tmp_path / "batch")
    row = result["summaries"][0]
    assert row["metadata_audit_status"] == "metadata_audited"
    assert row["raw_read_status"] == "read_integrity_failed"
    assert row["overall_status"] == "read_integrity_failed"
    assert result["metadata_audited_count"] == 1
    assert result["metadata_error_count"] == 0
    assert result["read_integrity_failed_count"] == 1
    assert result["error_count"] == 1


def test_select_rows_rejects_duplicate_or_out_of_range_indices():
    assert select_row_indices(10) == [0, 5, 9]
    assert select_row_indices(3, [2, 0]) == [2, 0]
    with pytest.raises(ValueError, match="duplicates"):
        select_row_indices(3, [1, 1])
    with pytest.raises(IndexError):
        select_row_indices(3, [3])


def test_batch_audit_writes_provenance_comparison_and_error_status(tmp_path: Path):
    data_dir = tmp_path / "tdms"
    valid = _write_tdms(
        data_dir / "a_valid.tdms",
        rows=2,
        channel_properties={
            f"Sweep{sweep}Step{step}": {"Wavelength (nm)": 1040.0 + step * 0.1}
            for sweep in range(2) for step in range(3)
        },
    )
    invalid = data_dir / "b_invalid.tdms"
    invalid.write_bytes(b"not a TDMS file")
    output_dir = tmp_path / "new_audit"
    result = audit_all_tdms(data_dir, output_dir)
    assert result["file_count"] == 2
    assert result["audited_count"] == 1
    assert result["error_count"] == 1
    assert result["summaries"][0]["wavelength_grade"] == "A"
    assert result["summaries"][0]["sampled_rows"] == [0, 1]
    assert result["summaries"][1]["status"] == "audit_error"
    assert result["summaries"][1]["error"]

    wavelength = json.loads((output_dir / valid.stem / "wavelength_evidence.json").read_text())
    assert len(wavelength["mapping"]) == 6
    assert valid.name in wavelength["mapping"][0]["source"]
    summary = json.loads((output_dir / invalid.stem / "audit_summary.json").read_text())
    assert summary["status"] == "audit_error"
    with (output_dir / "cross_file_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["file_name"] for row in rows] == [valid.name, invalid.name]
    assert rows[1]["error"]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit_all_tdms(data_dir, output_dir)

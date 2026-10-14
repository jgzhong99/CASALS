from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nptdms = pytest.importorskip("nptdms")
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

from research.tdms_audit import (
    analyze_channel_layout,
    discover_tdms_files,
    metadata_inventory,
    read_waveform_row,
    select_row_indices,
    select_tdms_file,
    wavelength_evidence,
)


def _write_tdms(path: Path, *, missing: tuple[int, int] | None = None,
                unequal_lengths: bool = False, explicit_wavelengths: bool = False,
                calibration: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_properties = {
        "# WVLSteps/Sweep": 3,
        "# Sweeps/ShiftCycle": 2,
        "Sweeps/File": 3,
        "Run duration (s)": 1.5,
    }
    if calibration:
        group_properties.update({
            "Laser wavelength calibration start (nm)": 1040.0,
            "Laser wavelength calibration increment (nm)": 0.1,
        })
    objects = [RootObject(properties={"Run label": "small test fixture"}),
               GroupObject("CASALS", properties=group_properties)]
    for sweep in range(2):
        for step in range(3):
            if missing == (sweep, step):
                continue
            length = 11 if unequal_lengths and (sweep, step) == (1, 2) else 12
            props = {"Wavelength (nm)": 1040.0 + 0.1 * step} if explicit_wavelengths else {}
            values = np.arange(length, dtype=np.int16) + 100 * sweep + 10 * step
            objects.append(ChannelObject("CASALS", f"Sweep{sweep}Step{step}", values, properties=props))
    with TdmsWriter(path) as writer:
        writer.write_segment(objects)
    return path


def _audit(path: Path):
    with nptdms.TdmsFile.read_metadata(path) as tdms:
        inventory = metadata_inventory(tdms)
    return inventory, analyze_channel_layout(inventory)


def test_metadata_properties_and_step_sweep_order(tmp_path: Path):
    path = _write_tdms(tmp_path / "small.tdms")
    inventory, layout = _audit(path)

    props = {(row["scope"], row["property_name"]): row for row in inventory["property_rows"]}
    assert props[("file", "Run label")]["raw_value"] == "small test fixture"
    assert props[("group", "# WVLSteps/Sweep")]["raw_value"] == 3
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
    assert layout["rows_per_file"] == 3
    assert layout["samples_per_row"] == 4


def test_layout_reports_missing_and_unequal_channels(tmp_path: Path):
    missing_inventory, missing_layout = _audit(_write_tdms(tmp_path / "missing.tdms", missing=(1, 2)))
    assert missing_layout["missing_step_sweep"] == [[1, 2]]
    assert not missing_layout["layout_complete"]

    _, unequal_layout = _audit(_write_tdms(tmp_path / "unequal.tdms", unequal_lengths=True))
    assert not unequal_layout["channel_lengths_consistent"]
    assert not unequal_layout["rows_file_divide_channel_length"]
    assert not unequal_layout["layout_complete"]


def test_scan_indices_do_not_create_physical_wavelengths(tmp_path: Path):
    inventory, layout = _audit(_write_tdms(tmp_path / "scan_only.tdms"))
    evidence = wavelength_evidence(inventory, layout)
    assert evidence["grade"] == "C"
    assert not evidence["physical_wavelength_available"]
    assert evidence["mapping"] == []
    assert any(row["property_name"] == "# WVLSteps/Sweep" for row in evidence["source_properties"])


def test_explicit_channel_wavelengths_and_calibration_mapping(tmp_path: Path):
    explicit_inventory, explicit_layout = _audit(
        _write_tdms(tmp_path / "explicit.tdms", explicit_wavelengths=True)
    )
    explicit = wavelength_evidence(explicit_inventory, explicit_layout)
    assert explicit["grade"] == "A"
    assert len(explicit["mapping"]) == 6
    assert explicit["mapping"][0]["wavelength_nm"] == 1040.0
    assert explicit["mapping"][-1]["wavelength_nm"] == 1040.2
    channel_property = next(row for row in explicit_inventory["property_rows"]
                            if row["scope"] == "channel" and row["property_name"] == "Wavelength (nm)")
    assert channel_property["raw_data_type"] == "float"
    assert channel_property["raw_value"] == 1040.0

    calibrated_inventory, calibrated_layout = _audit(
        _write_tdms(tmp_path / "calibrated.tdms", calibration=True)
    )
    calibrated = wavelength_evidence(calibrated_inventory, calibrated_layout)
    assert calibrated["grade"] == "B"
    assert len(calibrated["mapping"]) == 6
    assert calibrated["mapping"][0]["wavelength_nm"] == 1040.0
    assert calibrated["mapping"][-1]["wavelength_nm"] == 1040.2


def test_file_selection_excludes_index_sidecars_and_requires_explicit_choice(tmp_path: Path):
    data_dir = tmp_path / "data"
    first = _write_tdms(data_dir / "a.tdms")
    _write_tdms(data_dir / "b.tdms")
    (data_dir / "a.tdms_index").write_bytes(b"index")
    assert discover_tdms_files(data_dir) == [first, data_dir / "b.tdms"]
    with pytest.raises(ValueError, match="choose one explicitly"):
        select_tdms_file(tmp_path, data_dir, None)
    assert select_tdms_file(tmp_path, data_dir, "data/a.tdms") == first.resolve()
    with pytest.raises(ValueError, match=r"Expected a \.tdms file"):
        select_tdms_file(tmp_path, data_dir, "a.tdms_index")
    single_dir = tmp_path / "single"
    only_file = _write_tdms(single_dir / "only.tdms")
    assert select_tdms_file(tmp_path, single_dir, None) == only_file.resolve()


def test_selected_rows_and_waveform_reads_use_only_requested_slice():
    class TrackingChannel:
        def __init__(self):
            self.requested = []

        def __getitem__(self, key):
            self.requested.append(key)
            return np.arange(12, dtype=np.int16)[key]

    channel = TrackingChannel()
    values = read_waveform_row(channel, row_index=2, samples_per_row=4, total_rows=3)
    assert np.array_equal(values, [8, 9, 10, 11])
    assert channel.requested == [slice(8, 12)]
    assert select_row_indices(10, None) == [0, 5, 9]
    assert select_row_indices(3, [2, 0]) == [2, 0]
    with pytest.raises(IndexError):
        select_row_indices(3, [3])

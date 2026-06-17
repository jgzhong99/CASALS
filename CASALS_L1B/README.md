# CASALS L1B Refh Workflow

CASALS L1B in this project is treated as a geolocated waveform product, not a traditional discrete-return point cloud. Each pulse has one official geolocated `refh` reference point defined by `refh_longitude`, `refh_latitude`, and `refh`, and that `refh` corresponds to the Rx waveform maximum-amplitude bin. `refh` is treated here as WGS84 ellipsoidal height unless metadata explicitly says otherwise.

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `export_refh_las.py` | Export Level-A refh LAS from L1B H5 without noise labeling. |
| 2 | `filter_refh_points.py` | Label likely noise and write raw, noise-labeled, and clean refh LAS products. |
| 3 | `make_refh_dsm.py` | Build a support-limited filled refh surface DSM plus a raw strict companion DSM. |
| 4 | `extract_refh_ground.py` | Derive tentative ground candidates and an IDW DTM from high-SNR refh points. |
| 5 | `download_3dep_lpc.py` | Query 3DEP LPC workunits from the refh footprint and clip EPT-derived LAS/LAZ. |
| 6 | `diagnose_3dep_offsets.py` | Diagnose multi-pair CASALS vs 3DEP vertical offsets without writing corrected LAS. |
| 7 | `view_refh_points.py` | Open3D visualization of refh points by SNR, amplitude, height, good_snr, or classification. |
| 8 | `animate_pushbroom.py` | Animate refh sweep/track pushbroom acquisition. |
| 9 | `animate_rx_waveforms.py` | Animate raw `rx_waveform` frames by explicit sweep/track indexing. |

Recommended order:
1. `export_refh_las.py`
2. `filter_refh_points.py`
3. `make_refh_dsm.py`
4. `download_3dep_lpc.py`
5. `diagnose_3dep_offsets.py`
6. `view_refh_points.py` or the animation scripts as needed

Default output convention:
- `outputs/<script_name>/` stores JSON, PNG, TIF, CSV, MP4, and markdown reports.
- `point_cloud_data/<script_name>/` stores LAS, LAZ, and PLY outputs.
- `download_3dep_lpc.py` writes manifests and sidecars under `outputs/download_3dep_lpc/` and clipped LAZ/LAS under `point_cloud_data/download_3dep_lpc/`.

Notes:
- `export_refh_las.py` writes an unclassified Level-A refh LAS. LAS classification is all `1`.
- `filter_refh_points.py` is where noise labeling happens. LAS class `7` means likely noise.
- `make_refh_dsm.py` produces a support-limited filled refh surface DSM plus a raw strict audit DSM, not a classified ground DEM.
- `extract_refh_ground.py` produces a tentative derived ground candidate product, not an official ground DEM.
- `download_3dep_lpc.py` writes EPT-derived clips, not archival source LAZ copies, and does not perform a vertical datum transform.
- `diagnose_3dep_offsets.py` is diagnosis only. CASALS `refh` versus 3DEP comparisons must explicitly consider vertical datum and terrestrial reference frame differences.

Notebook helpers:
- `notebooks/01_l1b_structure_and_refh.ipynb`
- `notebooks/02_refh_quality_and_dsm.ipynb`
- `notebooks/03_3dep_offset_diagnosis.ipynb`

Do not use scripts under `archive/` as current entry points unless you are tracing pre-cleanup history.

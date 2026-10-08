# CASALS L1B pulse, waveform, and geolocation interpretation notes

Date: 2026-06-16  
Context: CASALS L1B H5 waveform data, `refh` reference-height point cloud, and feasibility of deriving multiple points from one pulse.

## 1. Bottom-line conclusions

The current CASALS L1B product should be understood as a **geolocated waveform product**, not as a conventional discrete-return point cloud.

For each pulse record, CASALS L1B provides:

- the full transmit waveform, `tx_waveform[pulse, tx_bin]`;
- the full receive waveform, `rx_waveform[pulse, rx_bin]`;
- one geolocated reference-height point, stored as `refh_longitude[pulse]`, `refh_latitude[pulse]`, and `refh[pulse]`;
- supporting per-pulse fields such as `refh_amp`, `refh_snr`, `refh_thres`, `good_snr`, `track_num`, `sweep_num`, `rwstart`, `rwstop`, instrument position, beam angle, and correction-related fields.

The key statement is:

> **CASALS L1B saves one official geolocated `refh` reference point per pulse. This point corresponds to the maximum-amplitude bin in the receive waveform. It is not an official multi-return decomposition, not a ground-classified point, and not a complete waveform-decomposed point cloud.**

Therefore:

- Directly exported CASALS L1B `refh` points should be called a **CASALS L1B `refh` reference-point cloud**.
- A point cloud produced by detecting additional peaks in `rx_waveform` should be called an **experimental waveform-peak point cloud**, unless the full CASALS geolocation processor, sensor pose, beam model, timing/range calibration, and correction model are used.

## 2. What is a pulse in CASALS L1B?

A pulse is a single laser shot with a unique timestamp and associated transmit and receive waveforms. CASALS documentation describes the organization as:

- **Pulse**: a single laser shot with unique timestamp and Tx/Rx waveforms.
- **Track**: a collection of pulses with the same wavelength/channel, illuminating the surface in the along-track direction.
- **Sweep**: a collection of coincident pulses spanning available wavelengths/channels, illuminating the surface in the across-track direction.

For the CASALS L1B examples we have been using, the H5 organization is typically:

```text
Dimensions:
  delta_time: 3,604,480 pulse records
  rx_bins: 2,728 receive waveform bins
  tx_bins: 80 transmit waveform bins
  phony_dim_0 / sweeps: 14,080 sweeps
  phony_dim_1 / tracks: 256 tracks
```

A complete rectangular organization can be interpreted as:

```text
14,080 sweeps × 256 tracks = 3,604,480 pulse records
```

However, robust code should use `sweep_num` and `track_num` to construct the sweep-track relationship instead of assuming that the H5 linear record order can always be reshaped directly.

## 3. What exactly is `refh`?

`refh` is the reference height associated with one receive-waveform bin: the bin with the maximum receive amplitude for that pulse. CASALS documentation describes it as the **geolocated height of the waveform bin with maximum amplitude**, in WGS84 ellipsoidal height convention.

For pulse `i`, the official reference point is:

```text
longitude_i = refh_longitude[i]
latitude_i  = refh_latitude[i]
height_i    = refh[i]
```

This point is one-dimensional over `delta_time`:

```text
refh.shape = (n_pulses,)
refh_longitude.shape = (n_pulses,)
refh_latitude.shape = (n_pulses,)
```

This means:

```text
one pulse → one official geolocated refh reference point
```

It does **not** mean:

```text
one pulse → one physical scatterer only
```

A single receive waveform may contain multiple peaks from canopy, ground, water, building surfaces, or mixed targets. `refh` only represents the maximum-amplitude reference bin.

## 4. Does L1B save multiple geolocated points for one pulse?

No. CASALS L1B saves the full waveform and one official `refh` reference point per pulse, but it does not provide an official list of multiple geolocated returns per pulse.

Specifically, it does not directly provide:

- one longitude/latitude/height triplet for every receive waveform bin;
- one longitude/latitude/height triplet for every local peak in `rx_waveform`;
- a waveform-decomposed multi-return point cloud;
- a ground/vegetation/building classified point cloud.

The L1B product gives enough information to inspect the waveform and derive experimental products, but not enough to treat additional waveform peaks as official CASALS geolocated returns without additional modeling.

## 5. Can multiple peaks be detected from one pulse?

Yes. The receive waveform `rx_waveform[i, :]` can be analyzed with standard full-waveform LiDAR methods:

```text
rx_waveform
→ background/noise estimation
→ smoothing or matched filtering
→ peak detection
→ peak fitting / waveform decomposition
→ peak bin or sub-bin peak position
```

This can produce multiple candidate peaks for a single pulse:

```text
pulse i:
  peak 1: bin b1, amplitude a1
  peak 2: bin b2, amplitude a2
  peak 3: bin b3, amplitude a3
  ...
```

But at this stage, these are **range-domain or waveform-domain peak candidates**, not yet strict 3D points in a mapping coordinate system.

## 6. What is needed to strictly geolocate multiple waveform peaks?

Strict multi-return geolocation requires a complete physical/geodetic model. Conceptually, each detected peak must be converted to range and then projected along the laser beam:

```text
P_return = P_sensor + range × unit_beam_vector
```

To do this rigorously, one needs:

- aircraft/sensor position at the pulse time;
- sensor attitude/orientation at the pulse time;
- scanner/beam pointing direction for the pulse/channel;
- lever-arm offsets between GNSS, IMU, and laser reference points;
- boresight calibration between sensor frames;
- transmit/receive timing calibration;
- bin-to-time and time-to-range conversion;
- speed-of-light and medium/correction assumptions;
- range bias correction;
- atmospheric delay correction;
- tide/geophysical corrections if they are part of the product definition;
- coordinate frame, vertical reference, and coordinate epoch.

Without these components, one can detect multiple peaks, but cannot assign them official-grade lon/lat/height coordinates.

## 7. Can L1B still support approximate multi-peak geolocation?

Yes, but only as an exploratory approximation.

CASALS L1B includes range-window start/stop variables and related geolocation fields, such as `rwstart`, `rwstop`, `rwstart_longitude`, `rwstart_latitude`, `rwstop_longitude`, `rwstop_latitude`, and supporting instrument/beam fields. One possible experimental approach is:

```text
1. Detect peaks in rx_waveform[i, :].
2. Assign each peak a bin index or sub-bin position.
3. Convert rwstart and rwstop positions to ECEF or local ENU.
4. Interpolate along the rwstart → rwstop segment to estimate a 3D position for each peak bin.
5. Convert the interpolated 3D point back to lon/lat/height.
6. Validate the maximum-amplitude peak against the official refh point.
```

The validation step is essential:

```text
If the interpolated maximum-amplitude-bin position closely reproduces refh,
then the range-window interpolation is at least geometrically consistent with refh.

If it does not reproduce refh,
then the approximation is not reliable and should not be used for multi-peak point generation.
```

Even if this validation passes, the result should still be described as experimental, because interpolation between `rwstart` and `rwstop` is not guaranteed to reproduce the internal CASALS geolocation processor for every bin and correction term.

## 8. Why a tilted beam matters

If the laser beam is not exactly vertical, different peaks along the same pulse do not share the same horizontal position. A later or earlier range bin moves along the beam ray, changing both height and horizontal coordinates.

Therefore, this is not rigorous:

```text
Use refh_longitude/refh_latitude for all peaks and only change height.
```

The more physically appropriate relationship is:

```text
Different peak bins → different ranges → different positions along the beam ray → different X/Y/Z.
```

This is the reason full sensor pose and beam geometry matter for strict multi-peak geolocation.

## 9. What can and cannot be claimed for the current L1B product?

### Safe claims

The following statements are safe:

- CASALS L1B contains full transmit and receive waveforms for each pulse.
- CASALS L1B contains one `refh` reference point per pulse.
- `refh` corresponds to the maximum-amplitude receive-waveform bin.
- A direct point cloud from `refh_longitude`, `refh_latitude`, and `refh` is a one-point-per-pulse reference-height point cloud.
- Additional waveform peaks can be detected experimentally from `rx_waveform`.
- Strict geolocation of those additional peaks requires a full sensor pose, beam, timing/range, calibration, and correction model.

### Claims to avoid

The following statements should be avoided unless additional CASALS processing documentation or code proves them:

- “CASALS L1B directly provides a multi-return point cloud.”
- “Every waveform peak has an official lon/lat/height in L1B.”
- “A peak detected from `rx_waveform` is automatically a geolocated return point.”
- “The `refh` point cloud is a ground point cloud.”
- “The `refh` point cloud is equivalent to 3DEP discrete-return LiDAR.”
- “Multiple peaks can be rigorously geolocated by changing only height while keeping `refh_longitude/refh_latitude` fixed.”

## 10. Recommended terminology

Use the following terms consistently:

```text
CASALS L1B waveform data
  The H5 product containing Tx/Rx waveforms and per-pulse reference-height variables.

CASALS L1B refh reference-point cloud
  One point per pulse from refh_longitude/refh_latitude/refh.

CASALS high-SNR refh points
  Subset of the refh reference-point cloud selected by refh_snr threshold.

CASALS tentative ground candidate
  A derived, non-official ground-like subset inferred from high-SNR and morphology/residual filtering.

CASALS experimental waveform-peak point cloud
  A non-official product produced by detecting additional peaks in rx_waveform and approximately geolocating them.

CASALS official multi-return point cloud
  Do not use this term unless CASALS provides or documents such a product.
```

## 11. Implications for 3DEP comparison

When comparing CASALS with 3DEP LiDAR:

1. CASALS `refh` points and 3DEP points are not the same product type.
2. 3DEP is a discrete-return classified point cloud; CASALS L1B `refh` is one maximum-amplitude reference point per pulse.
3. High-SNR CASALS `refh` points may be ground-like in many scenes, but this must be validated against 3DEP class-2 ground or another ground reference.
4. A waveform-derived multi-peak product from CASALS L1B should not be treated as an official equivalent of 3DEP returns without rigorous geolocation and validation.
5. Vertical datum/frame differences must be handled separately from waveform semantics.

## 12. Minimal validation recommended before using experimental multi-peak points

Before using any experimental multi-peak reconstruction from L1B, run these checks:

1. **Max-peak reproduction test**  
   Reconstruct the 3D position of the maximum-amplitude bin and compare it with `refh_longitude/refh_latitude/refh`.

2. **Bin-to-height monotonicity check**  
   Verify that interpolated bin height changes smoothly and physically along the range window.

3. **Beam-tilt sanity check**  
   Confirm that horizontal displacement per bin is plausible given beam angle and bin size.

4. **3DEP or independent surface comparison**  
   Compare high-confidence peaks with a transformed 3DEP ground/surface model.

5. **Correction sensitivity test**  
   Test whether range bias, atmospheric delay, tide, and other correction fields materially change the peak height.

6. **Scene-class sensitivity**  
   Evaluate separately over water, bare ground, vegetation, buildings, and mixed edges.

Only after these checks should experimental multi-peak products be used for interpretation.

## 13. Suggested wording for a manuscript or technical note

A rigorous wording is:

> CASALS L1B provides full transmit and receive waveforms for each laser pulse, together with one geolocated reference-height point per pulse. This reference point, `refh`, corresponds to the maximum-amplitude bin in the receive waveform and is stored with `refh_longitude` and `refh_latitude`. Therefore, the direct point cloud derived from L1B is a one-reference-point-per-pulse product, not an official multi-return or ground-classified LiDAR point cloud. Additional peaks may be extracted from the receive waveform, but rigorous geolocation of those peaks requires the complete sensor pose, beam geometry, timing/range calibration, and correction model. Without that information and validation, any multi-peak reconstruction should be treated as an experimental waveform-peak product.

## 14. Source notes

Primary CASALS source:

- CASALS L1B Waveform Data Tutorial, CryoCloud. This source documents the L1B product, pulse/track/sweep organization, waveform dimensions, `refh` definition, and key variables.  
  URL: https://book.cryointhecloud.com/l1b-waveforms-tutorial/

Supporting conceptual sources:

- NEON, The Basics of LiDAR. This source explains discrete-return and waveform-related LiDAR concepts at a general level.  
  URL: https://www.neonscience.org/resources/learning-hub/tutorials/lidar-basics

- Ayman Habib laser scanning georeferencing material. This source summarizes that georeferencing relates sensor and ground coordinate systems and depends on position/orientation information of the LiDAR unit or laser beam.  
  URL: https://engineering.purdue.edu/CCE/Academics/Groups/Geomatics/DPRG/Courses_Materials/LaserScanning_2019Spring/AKAM_2_Georeferencing.pdf

- Full-waveform LiDAR literature and tutorials. These sources support the distinction between waveform-domain peak detection/decomposition and geolocated discrete-return point production.  
  Example URL: https://iforest.sisef.org/contents/?id=ifor0562-004


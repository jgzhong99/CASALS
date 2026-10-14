"""Coordinate and geometric utilities for CASALS 3D Sweep Animator.

Handles:
- WGS84 Geodetic (EPSG:4979) <-> ECEF (EPSG:4978)
- ECEF <-> Local ENU (East-North-Up) relative to fixed origin
- H1 / H2 bin-to-XYZ mapping
- Aircraft schematic heading estimation from trajectory
- Sweep cross-track direction and meter projection
"""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np
# WGS84 ellipsoid parameters (EPSG:4979 / EPSG:4978)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2
WGS84_EP2 = (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)


def geodetic_to_ecef(lon: Any, lat: Any, height: Any) -> np.ndarray:
    """Convert WGS84 lon, lat (degrees) and ellipsoidal height (meters) to ECEF meters.

    Thread-safe, vectorized pure NumPy implementation.
    Returns an array of shape (..., 3).
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    h_arr = np.asarray(height, dtype=np.float64)

    lon_r = np.deg2rad(lon_arr)
    lat_r = np.deg2rad(lat_arr)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    N = WGS84_A / np.sqrt(np.maximum(1.0 - WGS84_E2 * (sin_lat ** 2), 1e-12))

    x = (N + h_arr) * cos_lat * cos_lon
    y = (N + h_arr) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h_arr) * sin_lat

    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr) & np.isfinite(h_arr)
    out = np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(np.float64, copy=False)
    if not np.all(valid):
        out[~valid] = np.nan
    return out


def ecef_to_geodetic(xyz: Any) -> np.ndarray:
    """Convert ECEF meter coordinates (..., 3) to WGS84 lon, lat (deg) and ellipsoidal height (m).

    Thread-safe, vectorized pure NumPy implementation using Bowring's algorithm.
    Returns an array of shape (..., 3) with [longitude, latitude, height].
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("ECEF coordinates must have a final dimension of size 3.")

    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]

    p = np.hypot(x, y)
    lon_r = np.arctan2(y, x)

    # Bowring's formula
    theta = np.arctan2(z * WGS84_A, np.maximum(p * WGS84_B, 1e-12))
    lat_r = np.arctan2(
        z + WGS84_EP2 * WGS84_B * (np.sin(theta) ** 3),
        p - WGS84_E2 * WGS84_A * (np.cos(theta) ** 3),
    )

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    N = WGS84_A / np.sqrt(np.maximum(1.0 - WGS84_E2 * (sin_lat ** 2), 1e-12))
    h = np.where(cos_lat != 0, p / cos_lat - N, np.abs(z) - WGS84_B)

    lon_deg = np.rad2deg(lon_r)
    lat_deg = np.rad2deg(lat_r)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    out = np.stack([lon_deg, lat_deg, h], axis=-1).astype(np.float64, copy=False)
    if not np.all(valid):
        out[~valid] = np.nan
    return out



class LocalENU:
    """Rigid transformation between ECEF (EPSG:4978) and local East-North-Up (ENU).

    The local origin (lon0, lat0, height0) is fixed for an entire granule.
    ENU coordinates are defined as:
    - X: East (meters)
    - Y: North (meters)
    - Z: Up (meters)
    """

    def __init__(self, origin_lon: float, origin_lat: float, origin_height: float) -> None:
        if not (np.isfinite(origin_lon) and np.isfinite(origin_lat) and np.isfinite(origin_height)):
            raise ValueError(f"Origin coordinates must be finite: ({origin_lon}, {origin_lat}, {origin_height})")

        self.origin_lon = float(origin_lon)
        self.origin_lat = float(origin_lat)
        self.origin_height = float(origin_height)

        # Origin in ECEF
        self.origin_ecef = geodetic_to_ecef(self.origin_lon, self.origin_lat, self.origin_height).reshape(3)

        # Orthonormal basis vectors for local ENU at origin (lon0, lat0)
        lon_r = np.deg2rad(self.origin_lon)
        lat_r = np.deg2rad(self.origin_lat)

        sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
        sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)

        # East unit vector in ECEF
        self.east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
        # North unit vector in ECEF
        self.north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
        # Up unit vector in ECEF
        self.up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)

        # Rotation matrix from ECEF difference to ENU: [East, North, Up]^T
        self.R = np.stack([self.east, self.north, self.up], axis=0)  # Shape (3, 3)

    def ecef_to_enu(self, xyz_ecef: Any) -> np.ndarray:
        """Transform ECEF coordinates (..., 3) to local ENU (..., 3)."""
        xyz = np.asarray(xyz_ecef, dtype=np.float64)
        if xyz.shape[-1] != 3:
            raise ValueError("xyz_ecef must have a final dimension of size 3.")
        diff = xyz - self.origin_ecef
        # diff @ R.T projects onto East, North, Up
        return np.matmul(diff, self.R.T)

    def enu_to_ecef(self, enu: Any) -> np.ndarray:
        """Transform local ENU coordinates (..., 3) to ECEF (..., 3)."""
        enu_arr = np.asarray(enu, dtype=np.float64)
        if enu_arr.shape[-1] != 3:
            raise ValueError("enu must have a final dimension of size 3.")
        # enu @ R + origin_ecef
        return np.matmul(enu_arr, self.R) + self.origin_ecef

    def geodetic_to_enu(self, lon: Any, lat: Any, height: Any) -> np.ndarray:
        """Convert WGS84 lon, lat, height to local ENU."""
        xyz = geodetic_to_ecef(lon, lat, height)
        return self.ecef_to_enu(xyz)

    def enu_to_geodetic(self, enu: Any) -> np.ndarray:
        """Convert local ENU coordinates to WGS84 lon, lat, height."""
        xyz = self.enu_to_ecef(enu)
        return ecef_to_geodetic(xyz)


def bin_to_xyz(
    start_xyz: Any,
    stop_xyz: Any,
    bins: Any,
    n_bins: int,
    *,
    convention: str = "H2",
) -> np.ndarray:
    """Map RX waveform bin index to 3D Cartesian coordinates along range window segment.

    Parameters
    ----------
    start_xyz : array-like, shape (..., 3)
        RWSTART Cartesian position (ECEF or ENU).
    stop_xyz : array-like, shape (..., 3)
        RWSTOP Cartesian position (ECEF or ENU).
    bins : array-like, shape (...)
        Zero-based bin indices in [0, n_bins - 1]. Fractional bins are supported.
    n_bins : int
        Total number of RX bins in the range window (typically 2728).
    convention : str, default 'H2'
        - 'H1': t = b / (n_bins - 1)  (endpoints are sample centers)
        - 'H2': t = (b + 0.5) / n_bins (window edges with centered bins; recommended for research)

    Returns
    -------
    np.ndarray, shape (..., 3)
        Interpolated 3D coordinates.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins}")

    a = np.asarray(start_xyz, dtype=np.float64)
    z = np.asarray(stop_xyz, dtype=np.float64)
    b = np.asarray(bins, dtype=np.float64)

    if a.shape[-1] != 3 or z.shape[-1] != 3:
        raise ValueError("Endpoints must have a final dimension of size 3.")

    convention_u = convention.upper()
    if convention_u == "H1":
        t = b / float(n_bins - 1)
    elif convention_u == "H2":
        t = (b + 0.5) / float(n_bins)
    else:
        raise ValueError(f"Unknown convention: {convention!r}. Choose 'H1' or 'H2'.")

    # Broadcast t with segment vector (z - a)
    t_expanded = np.expand_dims(t, axis=-1)
    return a + t_expanded * (z - a)


def compute_trajectory_headings(trajectory_enu: np.ndarray) -> np.ndarray:
    """Compute schematic aircraft heading (degrees, clockwise from North) along trajectory.

    Parameters
    ----------
    trajectory_enu : np.ndarray, shape (N, 3)
        Sequential instrument ENU coordinates.

    Returns
    -------
    np.ndarray, shape (N,)
        Heading angles in degrees in [0, 360).
    """
    n = len(trajectory_enu)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    east = trajectory_enu[:, 0]
    north = trajectory_enu[:, 1]

    # Compute differences along trajectory
    de = np.empty(n, dtype=np.float64)
    dn = np.empty(n, dtype=np.float64)

    # First point: forward difference
    de[0] = east[1] - east[0]
    dn[0] = north[1] - north[0]

    # Last point: backward difference
    de[-1] = east[-1] - east[-2]
    dn[-1] = north[-1] - north[-2]

    # Interior points: central difference
    if n > 2:
        de[1:-1] = (east[2:] - east[:-2]) * 0.5
        dn[1:-1] = (north[2:] - north[:-2]) * 0.5

    # Aviation heading: clockwise from North (0 = North, 90 = East, 180 = South, 270 = West)
    # arctan2(de, dn) gives angle from North in radians
    angles_rad = np.arctan2(de, dn)
    angles_deg = (np.rad2deg(angles_rad) + 360.0) % 360.0
    return angles_deg


def estimate_cross_track_direction(
    refh_enu_xy: np.ndarray,
    track_nums: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, bool, str]:
    """Estimate the cross-track horizontal direction unit vector and center for a sweep.

    Parameters
    ----------
    refh_enu_xy : np.ndarray, shape (M, 2)
        Local ENU (East, North) coordinates of valid Refh points in current sweep.
    track_nums : np.ndarray, shape (M,)
        Track numbers for the corresponding points.

    Returns
    -------
    unit_vector : np.ndarray of shape (2,) or None
        Normalized 2D vector along increasing track direction.
    center_xy : np.ndarray of shape (2,) or None
        Reference center point (East, North).
    success : bool
        Whether a reliable cross-track axis was found.
    reason : str
        Diagnostic description or reason for fallback.
    """
    if refh_enu_xy.shape[0] < 4:
        return None, None, False, "Fewer than 4 valid Refh points in sweep."

    xy = np.asarray(refh_enu_xy, dtype=np.float64)
    tracks = np.asarray(track_nums, dtype=np.float64)

    valid = np.isfinite(xy).all(axis=1) & np.isfinite(tracks)
    xy = xy[valid]
    tracks = tracks[valid]

    if xy.shape[0] < 4:
        return None, None, False, "Fewer than 4 finite Refh points."

    center = np.mean(xy, axis=0)
    xy_centered = xy - center

    # Covariance matrix for PCA
    cov = (xy_centered.T @ xy_centered) / float(len(xy))
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Principal direction corresponds to largest eigenvalue
    v = eigvecs[:, -1]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        return None, None, False, "Degenerate covariance in Refh ENU coordinates."

    v = v / norm_v

    # Check track span and spatial span
    proj = xy_centered @ v
    span_m = float(np.ptp(proj))
    track_span = float(np.ptp(tracks))

    if span_m < 5.0 or track_span < 10:
        return None, None, False, f"Insufficient spatial span ({span_m:.1f} m) or track span ({track_span})."

    # Ensure v points in the direction of increasing track number
    track_centered = tracks - np.mean(tracks)
    cov_track_proj = float(np.dot(track_centered, proj))
    if cov_track_proj < 0:
        v = -v
        proj = -proj

    # Check correlation between track number and projection
    corr = np.corrcoef(tracks, proj)[0, 1]
    if not np.isfinite(corr) or abs(corr) < 0.6:
        return None, None, False, f"Low track-to-spatial correlation ({corr:.2f})."

    return v, center, True, f"OK (span={span_m:.1f}m, corr={corr:.3f})"


def project_cross_track(
    points_enu_xy: np.ndarray,
    unit_vector: np.ndarray,
    center_xy: np.ndarray,
) -> np.ndarray:
    """Project ENU horizontal coordinates onto the cross-track unit vector relative to center.

    Parameters
    ----------
    points_enu_xy : np.ndarray, shape (..., 2)
        (East, North) coordinates.
    unit_vector : np.ndarray, shape (2,)
        Cross-track direction unit vector.
    center_xy : np.ndarray, shape (2,)
        Cross-track reference center.

    Returns
    -------
    np.ndarray, shape (...)
        Cross-track distances in meters.
    """
    diff = points_enu_xy - center_xy
    return np.matmul(diff, unit_vector)

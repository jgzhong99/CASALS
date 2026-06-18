"""Visualize a downloaded 3DEP LPC clip in Open3D using LAS classification colors.

Scientific meaning:
    Each displayed point is one point from a downloaded 3DEP LPC LAS/LAZ clip.

Outputs:
    An interactive Open3D view and console summaries only.

This script does not:
    - alter the source LAS/LAZ file,
    - merge multiple clips into one dataset,
    - write any derivative point-cloud products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import laspy
except Exception as exc:  # pragma: no cover
    raise ImportError("laspy is required. Install with: conda install -c conda-forge laspy lazrs") from exc


@dataclass(frozen=True)
class Config:
    input_las_path: Path
    output_dir: Path = Path("./outputs/view_3dep_lpc_open3d")
    save_legend_png: bool = True
    legend_png_name: str = "classification_color_legend.png"
    max_display_points: int = 800_000
    random_seed: int = 42
    center_xy_for_view: bool = True
    center_z_for_view: bool = True
    z_scale_for_view: float = 1.0
    point_size: float = 2.0
    show_coordinate_frame: bool = True
    coordinate_frame_size_m: float = 50.0
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    read_chunk_size: int = 2_000_000


@dataclass(frozen=True)
class SampledPointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray


@dataclass(frozen=True)
class LasViewSummary:
    input_path: Path
    point_count_total: int
    point_count_valid_xyz: int
    point_count_sampled: int
    las_version: str
    point_format_id: int
    classification_counts: dict[int, int]
    unknown_class_codes: list[int]
    crs_text: str
    crs_epsg: Optional[int]
    crs_name: str
    sampled_cloud: SampledPointCloud


CLASS_LABEL_MAP: dict[int, str] = {
    0: "never classified / unknown",
    1: "processed / unclassified",
    2: "ground / bare earth",
    3: "low vegetation",
    4: "medium vegetation",
    5: "high vegetation",
    6: "building",
    7: "low noise",
    9: "water",
    17: "bridge deck",
    18: "high noise",
    20: "ignored ground",
    21: "snow",
    22: "temporal exclusion",
}
CLASS_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.20, 0.20, 0.20),  # never classified
    1: (0.35, 0.52, 0.72),  # unclassified / processed
    2: (0.15, 0.75, 0.20),  # ground
    3: (0.55, 0.90, 0.35),  # low vegetation
    4: (0.25, 0.70, 0.25),  # medium vegetation
    5: (0.05, 0.45, 0.10),  # high vegetation
    6: (0.78, 0.48, 0.18),  # building
    7: (0.92, 0.15, 0.10),  # low noise
    9: (0.05, 0.78, 0.88),  # water
    17: (0.62, 0.32, 0.82),  # bridge deck
    18: (1.00, 0.05, 0.05),  # high noise
    20: (0.98, 0.82, 0.10),  # ignored ground
    21: (1.00, 1.00, 1.00),  # snow
    22: (0.92, 0.12, 0.75),  # temporal exclusion
}
FALLBACK_CLASS_COLOR = (0.52, 0.47, 0.62)
SCRIPT_DIR = Path(__file__).resolve().parent


def validate_config(cfg: Config) -> None:
    if not str(cfg.input_las_path).strip():
        raise ValueError("input_las_path must be a non-empty .las or .laz path.")
    if Path(cfg.input_las_path).suffix.lower() not in {".las", ".laz"}:
        raise ValueError("input_las_path must end with .las or .laz.")
    if not str(cfg.legend_png_name).strip():
        raise ValueError("legend_png_name must be non-empty.")
    if cfg.max_display_points <= 0:
        raise ValueError("max_display_points must be > 0.")
    if cfg.read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be > 0.")
    if len(cfg.background_rgb) != 3:
        raise ValueError("background_rgb must contain exactly 3 values.")
    for value in cfg.background_rgb:
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("background_rgb values must be finite numbers in [0, 1].")


def resolve_path_against_script_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    fallback = SCRIPT_DIR / path
    return fallback if fallback.exists() else path


def resolve_input_las_path(cfg: Config) -> Path:
    path = resolve_path_against_script_dir(Path(cfg.input_las_path))
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Configured input_las_path does not exist: {path}")
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Configured input_las_path must end with .las or .laz: {path}")
    return path


def resolve_output_dir(cfg: Config) -> Path:
    return resolve_path_against_script_dir(Path(cfg.output_dir))


def safe_parse_crs(header) -> tuple[str, Optional[int], str]:
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None

    if crs is None:
        return "CRS unavailable", None, ""

    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None

    try:
        text = crs.to_string()
    except Exception:
        text = "CRS parsed but to_string() failed"

    try:
        name = crs.name or ""
    except Exception:
        name = ""
    return text, epsg, name


def update_classification_counts(counts: dict[int, int], cls: np.ndarray) -> None:
    unique, count = np.unique(cls, return_counts=True)
    for key, value in zip(unique.tolist(), count.tolist()):
        counts[int(key)] = counts.get(int(key), 0) + int(value)


def merge_reservoir(
    keys_a: np.ndarray,
    points_a: np.ndarray,
    cls_a: np.ndarray,
    keys_b: np.ndarray,
    points_b: np.ndarray,
    cls_b: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keys_a.size == 0:
        merged_keys = keys_b
        merged_points = points_b
        merged_cls = cls_b
    elif keys_b.size == 0:
        merged_keys = keys_a
        merged_points = points_a
        merged_cls = cls_a
    else:
        merged_keys = np.concatenate([keys_a, keys_b])
        merged_points = np.vstack([points_a, points_b])
        merged_cls = np.concatenate([cls_a, cls_b])

    if merged_keys.size <= max_points:
        return merged_keys, merged_points, merged_cls

    keep = np.argpartition(merged_keys, -max_points)[-max_points:]
    return merged_keys[keep], merged_points[keep], merged_cls[keep]


def sampled_points_from_las(path: Path, cfg: Config) -> LasViewSummary:
    classification_counts: dict[int, int] = {}
    sampled_keys = np.empty(0, dtype=np.float64)
    sampled_points = np.empty((0, 3), dtype=np.float64)
    sampled_cls = np.empty(0, dtype=np.uint8)
    total_points = 0
    valid_xyz_points = 0
    rng = np.random.default_rng(cfg.random_seed)

    try:
        reader = laspy.open(path)
    except Exception as exc:
        msg = f"Failed to open LAS/LAZ file: {path}\n{type(exc).__name__}: {exc}"
        if path.suffix.lower() == ".laz":
            msg += "\nLAZ reading may require lazrs. Install with: conda install -c conda-forge lazrs"
        raise RuntimeError(msg) from exc

    with reader:
        header = reader.header
        crs_text, crs_epsg, crs_name = safe_parse_crs(header)
        las_version = str(header.version)
        point_format_id = int(header.point_format.id)

        for chunk in reader.chunk_iterator(cfg.read_chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            try:
                cls = np.asarray(chunk.classification, dtype=np.uint8)
            except Exception as exc:
                raise RuntimeError(
                    "LAS classification field could not be read; this script requires classification."
                ) from exc

            total_points += int(x.size)
            update_classification_counts(classification_counts, cls)

            valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            n_valid = int(np.sum(valid_mask))
            valid_xyz_points += n_valid
            if n_valid == 0:
                continue

            xyz = np.column_stack([x[valid_mask], y[valid_mask], z[valid_mask]])
            cls_valid = cls[valid_mask]
            keys = rng.random(n_valid)

            sampled_keys, sampled_points, sampled_cls = merge_reservoir(
                sampled_keys,
                sampled_points,
                sampled_cls,
                keys,
                xyz,
                cls_valid,
                cfg.max_display_points,
            )

    if total_points <= 0:
        raise RuntimeError(f"Input LAS/LAZ file has no points: {path}")
    if valid_xyz_points <= 0 or sampled_points.size == 0:
        raise RuntimeError(f"Input LAS/LAZ file has no finite XYZ points for visualization: {path}")

    order = np.argsort(sampled_keys)
    sampled_points = sampled_points[order]
    sampled_cls = sampled_cls[order]

    known_codes = set(CLASS_COLOR_MAP)
    unknown_codes = sorted(code for code in classification_counts if code not in known_codes)

    return LasViewSummary(
        input_path=path,
        point_count_total=total_points,
        point_count_valid_xyz=valid_xyz_points,
        point_count_sampled=int(sampled_points.shape[0]),
        las_version=las_version,
        point_format_id=point_format_id,
        classification_counts=dict(sorted(classification_counts.items())),
        unknown_class_codes=unknown_codes,
        crs_text=crs_text,
        crs_epsg=crs_epsg,
        crs_name=crs_name,
        sampled_cloud=SampledPointCloud(
            x=sampled_points[:, 0].astype(np.float64),
            y=sampled_points[:, 1].astype(np.float64),
            z=sampled_points[:, 2].astype(np.float64),
            classification=sampled_cls.astype(np.uint8),
        ),
    )


def classification_colors(classification: np.ndarray) -> np.ndarray:
    cls = np.asarray(classification, dtype=np.uint8)
    colors = np.tile(np.asarray(FALLBACK_CLASS_COLOR, dtype=np.float64), (cls.size, 1))
    for code, color in CLASS_COLOR_MAP.items():
        colors[cls == code] = np.asarray(color, dtype=np.float64)
    return colors


def color_to_text(color: tuple[float, float, float]) -> str:
    r = int(round(float(color[0]) * 255.0))
    g = int(round(float(color[1]) * 255.0))
    b = int(round(float(color[2]) * 255.0))
    return f"rgb({r:>3}, {g:>3}, {b:>3})"


def print_color_legend() -> None:
    print("Color legend:")
    for code in sorted(CLASS_COLOR_MAP):
        label = CLASS_LABEL_MAP.get(code, "unlabeled")
        print(f"  class {code:>2}: {color_to_text(CLASS_COLOR_MAP[code])} -> {label}")
    print(f"  other   : {color_to_text(FALLBACK_CLASS_COLOR)} -> fallback for unmapped class codes")
    print()


def write_color_legend_png(cfg: Config) -> Optional[Path]:
    if not cfg.save_legend_png:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Rectangle  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Legend PNG skipped: matplotlib unavailable ({type(exc).__name__}: {exc})")
        return None

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / cfg.legend_png_name

    entries = [(code, CLASS_COLOR_MAP[code], CLASS_LABEL_MAP.get(code, "unlabeled")) for code in sorted(CLASS_COLOR_MAP)]
    entries.append((-1, FALLBACK_CLASS_COLOR, "fallback for unmapped class codes"))

    fig_height = max(5.0, 0.55 * len(entries) + 0.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(entries))
    ax.axis("off")

    for row, (code, color, label) in enumerate(entries):
        y = len(entries) - row - 0.85
        ax.add_patch(Rectangle((0.4, y), 1.0, 0.6, facecolor=color, edgecolor="black", linewidth=0.6))
        code_text = "other" if code < 0 else f"class {code}"
        ax.text(1.7, y + 0.3, code_text, va="center", ha="left", fontsize=11, fontweight="bold")
        ax.text(3.1, y + 0.3, label, va="center", ha="left", fontsize=11)

    ax.set_title("3DEP LAS Classification Color Legend", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def centered_points_for_view(summary: LasViewSummary, cfg: Config) -> tuple[np.ndarray, tuple[float, float, float]]:
    x = summary.sampled_cloud.x
    y = summary.sampled_cloud.y
    z = summary.sampled_cloud.z

    x0 = float(np.nanmedian(x)) if cfg.center_xy_for_view else 0.0
    y0 = float(np.nanmedian(y)) if cfg.center_xy_for_view else 0.0
    z0 = float(np.nanmedian(z)) if cfg.center_z_for_view else 0.0

    points = np.column_stack([
        x - x0,
        y - y0,
        (z - z0) * float(cfg.z_scale_for_view),
    ]).astype(np.float64)
    return points, (x0, y0, z0)


def print_summary(summary: LasViewSummary, cfg: Config, center_xyz: tuple[float, float, float]) -> None:
    print("=" * 88)
    print("3DEP LPC Open3D visualization")
    print("=" * 88)
    print(f"Input file: {summary.input_path.resolve()}")
    print(f"LAS version: {summary.las_version}")
    print(f"Point format ID: {summary.point_format_id}")
    print(f"Total points: {summary.point_count_total:,}")
    print(f"Finite XYZ points: {summary.point_count_valid_xyz:,}")
    print(f"Sampled display points: {summary.point_count_sampled:,} (max={cfg.max_display_points:,})")
    print(f"CRS: {summary.crs_text}")
    if summary.crs_epsg is not None:
        print(f"CRS EPSG: {summary.crs_epsg}")
    if summary.crs_name:
        print(f"CRS name: {summary.crs_name}")
    print(
        "View center:"
        f" x={center_xyz[0]:.3f}, y={center_xyz[1]:.3f}, z={center_xyz[2]:.3f}"
        f" (z_scale_for_view={cfg.z_scale_for_view})"
    )
    print()
    print("Classification counts:")
    for code, count in summary.classification_counts.items():
        frac = count / max(summary.point_count_total, 1)
        label = CLASS_LABEL_MAP.get(code, "unmapped")
        print(f"  class {code:>2}: {count:>12,} ({frac:7.3%})  {label}")
    if summary.unknown_class_codes:
        print()
        print("Unknown class codes using fallback color:", ", ".join(str(v) for v in summary.unknown_class_codes))
    print()
    print_color_legend()


def visualize_open3d(summary: LasViewSummary, cfg: Config) -> None:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "open3d is required for visualization. Install with: conda install -c conda-forge open3d"
        ) from exc

    points, center_xyz = centered_points_for_view(summary, cfg)
    colors = classification_colors(summary.sampled_cloud.classification)
    legend_path = write_color_legend_png(cfg)
    print_summary(summary, cfg, center_xyz)
    if legend_path is not None:
        print(f"Legend PNG: {legend_path.resolve()}")
        print()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]
    if cfg.show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(cfg.coordinate_frame_size_m))
        geometries.append(frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3DEP LPC classification view", width=1400, height=950)
    for geometry in geometries:
        vis.add_geometry(geometry)

    opt = vis.get_render_option()
    opt.point_size = float(cfg.point_size)
    opt.background_color = np.asarray(cfg.background_rgb, dtype=np.float64)

    print("Open3D controls:")
    print("  Mouse drag: rotate")
    print("  Mouse wheel: zoom")
    print("  Shift/Ctrl + drag: pan")
    print("  Press 'Q' or close the window to exit")
    print()

    vis.run()
    vis.destroy_window()


def main() -> None:
    cfg = Config(
        # Set this to the exact 3DEP LAS/LAZ clip you want to view.
        #input_las_path=Path("./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz"),
        #input_las_path=Path("./outputs/transfer_3dep_labels_to_casals_refh/casals_l1b_single_casals_3dep_pseudolabeled_aligned.las"),
        #input_las_path=Path("./point_cloud_data/classify_refh_3dep_like_rules/casals_l1b_20241112T165718_001_02_3dep_like_rules_classified.las"),
        input_las_path=Path("./outputs/transfer_3dep_labels_to_casals_refh_multi/pair1_md_casals_3dep_pseudolabeled_aligned.las"),
        output_dir=Path("./outputs/view_3dep_lpc_open3d"),
        save_legend_png=True,
        legend_png_name="classification_color_legend.png",
        max_display_points=800_000,
        random_seed=42,
        center_xy_for_view=True,
        center_z_for_view=True,
        z_scale_for_view=1.0,
        point_size=2.0,
        show_coordinate_frame=True,
        coordinate_frame_size_m=50.0,
        background_rgb=(0.0, 0.0, 0.0),
        read_chunk_size=2_000_000,
    )

    validate_config(cfg)
    input_path = resolve_input_las_path(cfg)
    summary = sampled_points_from_las(input_path, cfg)
    visualize_open3d(summary, cfg)


if __name__ == "__main__":
    main()

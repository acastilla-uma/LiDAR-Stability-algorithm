#!/usr/bin/env python3
"""
Inspect a small LiDAR map slice with truck route and terrain features.

This diagnostic tool is intended to quickly validate whether traversability
features look coherent on a local route portion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.visualization.visualize_route_lidar import (
    deduplicate_route,
    find_laz_tiles,
    load_laz_as_points,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local route + LiDAR slice with computed terrain features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapmatch", required=True, help="Map-matched CSV with x_utm/y_utm and pre-computed terrain features")
    parser.add_argument("--laz-dir", default=None, help="Directory with LAZ files")
    parser.add_argument("--output", default="output/route_terrain_slice.html", help="Output HTML path")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV with inspected points + features")

    # Visualization options only
    parser.add_argument("--search-radius", type=float, default=100.0, help="Search radius to filter LiDAR points around route (m)")
    parser.add_argument("--sampling", type=int, default=1, help="Process every nth route point")

    # Slice controls
    parser.add_argument("--start-index", type=int, default=0, help="Start index in sampled route")
    parser.add_argument("--max-route-points", type=int, default=40, help="Maximum route points to inspect")
    parser.add_argument("--padding", type=float, default=80.0, help="Extra map padding around selected slice (m)")
    parser.add_argument("--max-points", type=int, default=120000, help="Maximum LiDAR points rendered")
    parser.add_argument(
        "--max-render-points",
        type=int,
        default=250000,
        help="Safety cap for 3D WebGL rendering. Cloud is downsampled if exceeded.",
    )

    parser.add_argument("--stability-col", default="si", help="Stability column for route coloring if present")
    parser.add_argument("--no-ground-filter", action="store_true", help="Do not filter non-ground LiDAR points")
    parser.add_argument("--print-rows", type=int, default=15, help="Number of rows printed to stdout")
    return parser.parse_args()


def _build_hover_text(df: pd.DataFrame, stability_col: str) -> list[str]:
    rows = []
    has_si = stability_col in df.columns
    for _, row in df.iterrows():
        parts = [
            f"<b>idx</b>: {int(row['slice_idx'])}",
            f"<b>x_utm</b>: {row['x_utm']:.2f}",
            f"<b>y_utm</b>: {row['y_utm']:.2f}",
            f"<b>phi_deg</b>: {row['phi_lidar_deg']:.2f}",
            f"<b>tri</b>: {row['tri']:.4f}",
            f"<b>ruggedness</b>: {row['ruggedness']:.4f}",
            f"<b>z_mean</b>: {row['z_mean']:.2f}",
            f"<b>n_points_used</b>: {int(row['n_points_used'])}",
        ]
        if has_si and pd.notna(row.get(stability_col, np.nan)):
            parts.insert(3, f"<b>{stability_col}</b>: {row[stability_col]:.3f}")
        rows.append("<br>".join(parts))
    return rows


def inspect_route_terrain_slice(args: argparse.Namespace) -> tuple[Path, pd.DataFrame]:
    mapmatch_path = Path(args.mapmatch)
    if not mapmatch_path.exists():
        raise FileNotFoundError(f"Map-matched file not found: {mapmatch_path}")

    project_root = Path(__file__).resolve().parents[3]
    laz_dir = Path(args.laz_dir) if args.laz_dir else project_root / "LiDAR-Maps" / "cnig"
    if not laz_dir.exists():
        raise FileNotFoundError(f"LAZ directory not found: {laz_dir}")

    df = pd.read_csv(mapmatch_path)
    required = {"x_utm", "y_utm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["x_utm", "y_utm"]).copy()
    if args.stability_col in df.columns:
        df_route = deduplicate_route(df, args.stability_col)
    else:
        df_route = df.drop_duplicates(subset=["x_utm", "y_utm"]).reset_index(drop=True)

    if args.sampling > 1:
        df_route = df_route.iloc[:: args.sampling].reset_index(drop=True)

    if df_route.empty:
        raise RuntimeError("No route points available after filtering/sampling")

    start = max(0, int(args.start_index))
    end = min(len(df_route), start + max(1, int(args.max_route_points)))
    inspected_df = df_route.iloc[start:end].copy().reset_index(drop=True)
    if inspected_df.empty:
        raise RuntimeError("Selected slice is empty; adjust --start-index / --max-route-points")

    # Add slice_idx column based on original row indices
    inspected_df.insert(0, "slice_idx", range(start, start + len(inspected_df)))

    # Verify that pre-computed features exist in CSV
    feature_cols = ["phi_lidar_deg", "tri", "ruggedness", "z_mean", "n_points_used"]
    missing_features = [col for col in feature_cols if col not in inspected_df.columns]
    if missing_features:
        raise ValueError(
            f"Missing pre-computed feature columns in CSV: {missing_features}. "
            f"Expected columns: {feature_cols}"
        )

    logger.info("Loaded %d route points with pre-computed features from CSV", len(inspected_df))

    x_min = float(inspected_df["x_utm"].min() - args.padding)
    x_max = float(inspected_df["x_utm"].max() + args.padding)
    y_min = float(inspected_df["y_utm"].min() - args.padding)
    y_max = float(inspected_df["y_utm"].max() + args.padding)

    laz_tiles = find_laz_tiles(x_min, y_min, x_max, y_max, laz_dir)
    if not laz_tiles:
        raise RuntimeError(f"No LAZ tiles found for selected slice in {laz_dir}")

    cloud = load_laz_as_points(
        tile_paths=laz_tiles,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        max_points=int(args.max_points),
        filter_ground=not args.no_ground_filter,
    )
    if len(cloud) == 0:
        raise RuntimeError("No LiDAR points loaded for selected slice")

    # Keep only points within search_radius from at least one route position.
    route_xy = inspected_df[["x_utm", "y_utm"]].to_numpy(dtype=float)
    tree = cKDTree(route_xy)
    nearest_dist, _ = tree.query(cloud[:, :2], k=1)
    in_radius = nearest_dist <= float(args.search_radius)
    cloud = cloud[in_radius]
    if len(cloud) == 0:
        raise RuntimeError(
            "No LiDAR points found inside search radius around selected route points. "
            "Increase --search-radius or choose another slice."
        )
    logger.info(
        "Cloud points kept inside radius %.2fm: %d",
        float(args.search_radius),
        len(cloud),
    )

    if len(cloud) < 500:
        logger.warning(
            "Very few LiDAR points are available inside radius %.2fm (%d points). "
            "Increase --search-radius or reduce --sampling if the cloud looks sparse.",
            float(args.search_radius),
            len(cloud),
        )

    max_render_points = max(10000, int(args.max_render_points))
    if len(cloud) > max_render_points:
        original_n = len(cloud)
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(original_n, size=max_render_points, replace=False)
        cloud = cloud[keep_idx]
        logger.info(
            "Downsampled cloud for 3D rendering: %d -> %d points (set --max-render-points to change)",
            original_n,
            max_render_points,
        )

    hover_text = _build_hover_text(inspected_df, args.stability_col)

    route_z = inspected_df["z_mean"].to_numpy(dtype=float)
    if np.any(~np.isfinite(route_z)):
        fallback_z = float(np.nanmean(cloud[:, 2]))
        route_z = np.where(np.isfinite(route_z), route_z, fallback_z)

    if len(cloud) < 500:
        cloud_marker_size = 5.5
        cloud_opacity = 0.95
    elif len(cloud) < 5000:
        cloud_marker_size = 3.5
        cloud_opacity = 0.8
    else:
        cloud_marker_size = 1.8
        cloud_opacity = 0.45

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode="markers",
            marker=dict(
                size=cloud_marker_size,
                color=cloud[:, 2],
                colorscale="Earth",
                showscale=True,
                colorbar=dict(title="Elevation (m)"),
                opacity=cloud_opacity,
            ),
            name="LiDAR",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if args.stability_col in inspected_df.columns:
        route_color = inspected_df[args.stability_col]
        color_title = args.stability_col
        colorscale = "RdYlGn"
        cmin = 0.0
        cmax = 1.0
    else:
        route_color = inspected_df["phi_lidar_deg"]
        color_title = "phi_lidar_deg"
        colorscale = "Turbo"
        cmin = float(np.nanpercentile(route_color, 2))
        cmax = float(np.nanpercentile(route_color, 98))

    fig.add_trace(
        go.Scatter3d(
            x=inspected_df["x_utm"],
            y=inspected_df["y_utm"],
            z=route_z,
            mode="lines+markers",
            line=dict(color="rgba(20,20,20,0.35)", width=2),
            marker=dict(
                size=5,
                color=route_color,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title=f"Route color ({color_title})", x=1.12),
            ),
            text=hover_text,
            hovertemplate="%{text}<br><b>z</b>: %{z:.2f}<extra></extra>",
            name="Route slice",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[inspected_df.iloc[0]["x_utm"]],
            y=[inspected_df.iloc[0]["y_utm"]],
            z=[route_z[0]],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[inspected_df.iloc[-1]["x_utm"]],
            y=[inspected_df.iloc[-1]["y_utm"]],
            z=[route_z[-1]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond"),
            name="End",
        )
    )

    title = (
        f"Route terrain slice | {mapmatch_path.name}<br>"
        f"<sub>points={len(inspected_df)} | search_radius={args.search_radius}m | sampling={args.sampling}</sub>"
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1350,
        height=850,
        hovermode="closest",
        scene=dict(
            xaxis_title="UTM X (m)",
            yaxis_title="UTM Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.95)),
        ),
    )

    out_html = Path(args.output)
    if not out_html.is_absolute():
        out_html = (project_root / out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)

    if args.output_csv:
        out_csv = Path(args.output_csv)
        if not out_csv.is_absolute():
            out_csv = (project_root / out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        inspected_df.to_csv(out_csv, index=False)

    return out_html, inspected_df


def main() -> int:
    args = parse_args()
    try:
        out_html, inspected_df = inspect_route_terrain_slice(args)

        show_cols = [
            "slice_idx",
            "x_utm",
            "y_utm",
            "phi_lidar_deg",
            "tri",
            "ruggedness",
            "z_mean",
            "n_points_used",
        ]
        if args.stability_col in inspected_df.columns:
            show_cols.insert(3, args.stability_col)

        print("\nRoute terrain slice preview:")
        print(inspected_df[show_cols].head(max(1, args.print_rows)).to_string(index=False))
        print(f"\nOK: HTML saved to {out_html}")
        return 0
    except Exception as exc:
        logger.error(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

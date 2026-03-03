#!/usr/bin/env python3
"""
Visualización 2D de ruta map-matched sobre nube LiDAR (LAZ).
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from lidar.laz_reader import LAZReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_laz_tiles(x_min: float, y_min: float, x_max: float, y_max: float, laz_dir: Path) -> list[Path]:
    tx_min = int(x_min // 1000)
    tx_max = int(x_max // 1000)
    ty_min = int(y_min // 1000) + 1
    ty_max = int(y_max // 1000) + 1

    tiles: list[Path] = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tiles.extend(laz_dir.glob(f"*_{tx}-{ty}_*.laz"))
            tiles.extend(laz_dir.glob(f"*_{tx}-{ty}.laz"))
    return sorted(set(tiles))


def load_laz_as_points(tile_paths: list[Path], x_min: float, y_min: float, x_max: float, y_max: float,
                       max_points: int = 200_000, filter_ground: bool = True) -> np.ndarray:
    margin = 50
    all_pts: list[np.ndarray] = []

    for path in tile_paths:
        try:
            logger.info(f"  Cargando LAZ: {path.name}")
            reader = LAZReader(str(path), filter_ground=filter_ground)
            pts = reader.extract_patch(
                x_center=(x_min + x_max) / 2,
                y_center=(y_min + y_max) / 2,
                radius_m=max(x_max - x_min, y_max - y_min) / 2 + margin,
            )
            if pts is not None and len(pts) > 0:
                mask = (
                    (pts[:, 0] >= x_min - margin) & (pts[:, 0] <= x_max + margin) &
                    (pts[:, 1] >= y_min - margin) & (pts[:, 1] <= y_max + margin)
                )
                pts = pts[mask]
                if len(pts) > 0:
                    all_pts.append(pts)
        except Exception as exc:
            logger.warning(f"  No se pudo cargar {path.name}: {exc}")

    if not all_pts:
        return np.empty((0, 3))

    combined = np.vstack(all_pts)
    if len(combined) > max_points:
        rng = np.random.default_rng(42)
        combined = combined[rng.choice(len(combined), size=max_points, replace=False)]
    return combined


def deduplicate_route(df: pd.DataFrame, stability_col: str, x_col: str = "x_utm", y_col: str = "y_utm") -> pd.DataFrame:
    df = df.copy()
    df["_x_r"] = df[x_col].round(5)
    df["_y_r"] = df[y_col].round(5)
    key = df["_x_r"].astype(str) + "_" + df["_y_r"].astype(str)
    mask = key != key.shift()
    return df[mask].drop(columns=["_x_r", "_y_r"]).reset_index(drop=True)


def colored_route_segments(x: np.ndarray, y: np.ndarray, values: np.ndarray, cmap_name: str = "RdYlGn"):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.colormaps.get_cmap(cmap_name)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
    lc.set_array(values[:-1])
    return lc, norm


def visualize_route_on_lidar(mapmatch_path: str, laz_dir: str = None, tif_dir: str = None, source: str = "auto",
                             max_points: int = 200_000, stability_col: str = "si", output_path: str = None,
                             filter_ground: bool = True, padding_m: float = 100.0,
                             show_coordinates: bool = False, coordinates_only: bool = False):
    mapmatch_path = Path(mapmatch_path)
    if not mapmatch_path.exists():
        raise FileNotFoundError(f"Map-matched file not found: {mapmatch_path}")

    project_root = Path(__file__).parent.parent.parent
    laz_dir = Path(laz_dir) if laz_dir else project_root / "LiDAR-Maps" / "cnig"

    df = pd.read_csv(mapmatch_path)
    required = {"x_utm", "y_utm", stability_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas requeridas ausentes: {missing}")

    df = df.dropna(subset=list(required))
    df_route = deduplicate_route(df, stability_col)

    x_route = df_route["x_utm"].values
    y_route = df_route["y_utm"].values
    si_route = df_route[stability_col].values

    if show_coordinates or coordinates_only:
        print(f"Inicio: ({x_route[0]:.3f}, {y_route[0]:.3f})")
        print(f"Fin: ({x_route[-1]:.3f}, {y_route[-1]:.3f})")

    if coordinates_only:
        return True

    x_min = x_route.min() - padding_m
    x_max = x_route.max() + padding_m
    y_min = y_route.min() - padding_m
    y_max = y_route.max() + padding_m

    laz_tiles = find_laz_tiles(x_min, y_min, x_max, y_max, laz_dir)
    if not laz_tiles:
        raise RuntimeError(f"No se encontraron tiles LAZ en: {laz_dir}")

    cloud = load_laz_as_points(laz_tiles, x_min, y_min, x_max, y_max, max_points=max_points,
                               filter_ground=filter_ground)
    if len(cloud) == 0:
        raise RuntimeError("No se pudieron cargar puntos LAZ")

    fig, ax = plt.subplots(figsize=(14, 10))
    z_vals = cloud[:, 2]
    z_norm = mcolors.Normalize(vmin=np.percentile(z_vals, 2), vmax=np.percentile(z_vals, 98))
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=z_vals, cmap="terrain", norm=z_norm,
                    s=0.3, alpha=0.5, linewidths=0, zorder=2, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, aspect=40).set_label("Elevación (m)")

    lc, norm_si = colored_route_segments(x_route, y_route, si_route)
    ax.add_collection(lc)
    sm = plt.cm.ScalarMappable(norm=norm_si, cmap="RdYlGn")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.05, aspect=40).set_label(f"Estabilidad ({stability_col})")

    ax.plot(x_route[0], y_route[0], "go", ms=8, zorder=6, label="Inicio")
    ax.plot(x_route[-1], y_route[-1], "rs", ms=8, zorder=6, label="Fin")
    ax.set_title(f"Ruta sobre LiDAR | {mapmatch_path.stem}")
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Figura guardada en: {out}")
    else:
        plt.show()

    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualizar ruta map-matched sobre LiDAR (2D)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mapmatch", required=True)
    parser.add_argument("--laz-dir", default=None)
    parser.add_argument("--tif-dir", default=None)
    parser.add_argument("--source", default="auto", choices=["auto", "laz", "tif"])
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--stability-col", default="si")
    parser.add_argument("--no-ground-filter", action="store_true")
    parser.add_argument("--padding", type=float, default=100.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-coordinates", action="store_true")
    parser.add_argument("--coordinates-only", action="store_true")
    args = parser.parse_args()

    try:
        visualize_route_on_lidar(
            mapmatch_path=args.mapmatch,
            laz_dir=args.laz_dir,
            tif_dir=args.tif_dir,
            source=args.source,
            max_points=args.max_points,
            stability_col=args.stability_col,
            output_path=args.output,
            filter_ground=not args.no_ground_filter,
            padding_m=args.padding,
            show_coordinates=args.show_coordinates,
            coordinates_only=args.coordinates_only,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"❌ ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualización 2D interactiva de ruta map-matched sobre nube LiDAR (LAZ).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

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


def visualize_route_on_lidar(mapmatch_path: str, laz_dir: str = None, tif_dir: str = None, source: str = "auto",
                             max_points: int = 200_000, stability_col: str = "si", output_path: str = None,
                             filter_ground: bool = True, padding_m: float = 100.0,
                             show_coordinates: bool = False, coordinates_only: bool = False,
                             show_terrain_features: bool = True):
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
    
    # Check for terrain features
    terrain_feature_cols = ['phi_lidar', 'phi_lidar_deg', 'tri', 'ruggedness', 'z_min', 'z_max', 'z_mean', 'z_std', 'z_range']
    available_features = [col for col in terrain_feature_cols if col in df_route.columns]
    if available_features and show_terrain_features:
        logger.info(f"Terrain features found: {available_features}")

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

    # Create interactive Plotly figure
    fig = go.Figure()
    
    # Add LiDAR cloud background
    z_vals = cloud[:, 2]
    z_min_val = np.percentile(z_vals, 2)
    z_max_val = np.percentile(z_vals, 98)
    
    fig.add_trace(go.Scatter(
        x=cloud[:, 0],
        y=cloud[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=z_vals,
            colorscale='Earth',
            cmin=z_min_val,
            cmax=z_max_val,
            showscale=True,
            colorbar=dict(title="Elevación (m)")
        ),
        name='LiDAR',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Build hover text for route points with terrain features
    hover_text = []
    for idx in range(len(df_route)):
        row = df_route.iloc[idx]
        text_parts = [
            f"<b>Índice</b>: {idx}",
            f"<b>X</b>: {row['x_utm']:.2f}",
            f"<b>Y</b>: {row['y_utm']:.2f}",
            f"<b>SI</b>: {row[stability_col]:.3f}",
        ]
        
        # Add terrain features if available
        if available_features and show_terrain_features:
            for feat in available_features:
                val = row[feat]
                if pd.notna(val):
                    if feat == 'phi_lidar_deg':
                        text_parts.append(f"<b>φ (°)</b>: {val:.2f}")
                    elif feat == 'phi_lidar':
                        text_parts.append(f"<b>φ (rad)</b>: {val:.4f}")
                    elif feat == 'tri':
                        text_parts.append(f"<b>TRI</b>: {val:.4f}")
                    elif feat == 'ruggedness':
                        text_parts.append(f"<b>Ruggedness</b>: {val:.4f}")
                    elif feat in ['z_min', 'z_max', 'z_mean', 'z_std', 'z_range']:
                        short_name = feat.replace('z_', 'Z_').upper()
                        text_parts.append(f"<b>{short_name}</b>: {val:.2f}m")
        
        hover_text.append("<br>".join(text_parts))
    
    # Add route segments with SI coloring
    # Create segments colored by SI value
    for i in range(len(x_route) - 1):
        color_si = si_route[i]
        # Map SI value (0-1) to RGB using RdYlGn colorscale
        # Red (low stability) = [1, 0, 0], Yellow (medium) = [1, 1, 0], Green (high) = [0, 1, 0]
        if color_si < 0.5:
            # Red to Yellow
            ratio = color_si * 2  # 0-1 within red-yellow range
            r, g, b = 1, ratio, 0
        else:
            # Yellow to Green
            ratio = (color_si - 0.5) * 2  # 0-1 within yellow-green range
            r, g, b = 1 - ratio, 1, 0
        
        rgb_color = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        
        fig.add_trace(go.Scatter(
            x=[x_route[i], x_route[i+1]],
            y=[y_route[i], y_route[i+1]],
            mode='lines',
            line=dict(color=rgb_color, width=4),
            hovertemplate='%{text}<extra></extra>',
            text=[hover_text[i], hover_text[i+1]],
            showlegend=False,
            name='Ruta'
        ))
    
    # Add route markers
    fig.add_trace(go.Scatter(
        x=x_route,
        y=y_route,
        mode='markers',
        marker=dict(
            size=6,
            color=si_route,
            colorscale='RdYlGn',
            cmin=0.0,
            cmax=1.0,
            showscale=True,
            colorbar=dict(title=f"Estabilidad ({stability_col})", x=1.12)
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Ruta',
        showlegend=True
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter(
        x=[x_route[0]],
        y=[y_route[0]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='Inicio',
        hovertext='Inicio',
        hoverinfo='text+x+y'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_route[-1]],
        y=[y_route[-1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='Fin',
        hovertext='Fin',
        hoverinfo='text+x+y'
    ))
    
    # Build title
    title_str = f"Ruta sobre LiDAR | {mapmatch_path.stem}"
    if available_features and show_terrain_features:
        title_str += f"<br><sub>Características: {', '.join(available_features[:5])}</sub>"
    
    fig.update_layout(
        title=title_str,
        xaxis_title="UTM X (m)",
        yaxis_title="UTM Y (m)",
        hovermode='closest',
        width=1400,
        height=900,
        template='plotly_white',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Change extension to .html if it's .png
        if str(out).endswith('.png'):
            out = out.with_suffix('.html')
        fig.write_html(out)
        print(f"✅ Visualización guardada en: {out}")
    else:
        fig.show()

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
    parser.add_argument("--show-terrain-features", action="store_true", default=True,
                       help="Display terrain features in visualization if available")
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
            show_terrain_features=args.show_terrain_features,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"❌ ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

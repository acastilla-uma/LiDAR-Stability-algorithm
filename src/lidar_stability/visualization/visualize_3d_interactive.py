#!/usr/bin/env python3
"""
VisualizaciÃ³n 3D interactiva de segmentos sobre nube LiDAR.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.visualization.visualize_route_lidar import find_laz_tiles, load_laz_as_points, deduplicate_route

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_segment_files(base_name: str, mapmatch_dir: Path) -> list[Path]:
    pattern = f"{base_name}_seg*.csv"
    files = sorted(mapmatch_dir.glob(pattern))
    base_file = mapmatch_dir / f"{base_name}.csv"
    if base_file.exists():
        files.insert(0, base_file)
    return files


def visualize_3d_interactive(base_name: str, mapmatch_dir: str = None, laz_dir: str = None,
                             points_sample: int = 50_000, stability_col: str = "si",
                             output_path: str = None, filter_ground: bool = True,
                             padding_m: float = 100.0, show_terrain_features: bool = True):
    project_root = Path(__file__).resolve().parents[3]
    if mapmatch_dir:
        mapmatch_dir = Path(mapmatch_dir)
    else:
        featured_dir = project_root / "Doback-Data" / "featured"
        fallback_mapmatched_dir = project_root / "Doback-Data" / "map-matched"
        mapmatch_dir = featured_dir

        if not list(featured_dir.glob(f"{base_name}*.csv")) and list(fallback_mapmatched_dir.glob(f"{base_name}*.csv")):
            mapmatch_dir = fallback_mapmatched_dir
    laz_dir = Path(laz_dir) if laz_dir else project_root / "LiDAR-Maps" / "cnig"

    segment_files = find_segment_files(base_name, mapmatch_dir)
    if not segment_files:
        raise FileNotFoundError(f"No se encontraron segmentos para '{base_name}' en {mapmatch_dir}")

    segments_data = []
    all_x, all_y = [], []
    
    # Define terrain feature columns
    terrain_feature_cols = ['phi_lidar', 'phi_lidar_deg', 'tri', 'ruggedness', 'z_min', 'z_max', 'z_mean', 'z_std', 'z_range']

    for seg_file in segment_files:
        df = pd.read_csv(seg_file)
        required = {"x_utm", "y_utm", stability_col}
        if not required.issubset(df.columns):
            continue
        df = df.dropna(subset=list(required))
        df = deduplicate_route(df, stability_col)
        if len(df) < 2:
            continue

        x = df["x_utm"].values
        y = df["y_utm"].values
        si = df[stability_col].values
        
        # Check for available terrain features
        available_features = [col for col in terrain_feature_cols if col in df.columns]
        
        seg_data = {"name": seg_file.stem, "x": x, "y": y, "si": si, "available_features": available_features}
        
        # Store terrain feature values if available
        if available_features and show_terrain_features:
            for feat in available_features:
                seg_data[feat] = df[feat].values
        
        segments_data.append(seg_data)
        all_x.extend(x)
        all_y.extend(y)

    if not segments_data:
        raise RuntimeError("No hay segmentos vÃ¡lidos para visualizar")

    all_x_arr = np.array(all_x)
    all_y_arr = np.array(all_y)

    x_min = all_x_arr.min() - padding_m
    x_max = all_x_arr.max() + padding_m
    y_min = all_y_arr.min() - padding_m
    y_max = all_y_arr.max() + padding_m

    laz_tiles = find_laz_tiles(x_min, y_min, x_max, y_max, laz_dir)
    if not laz_tiles:
        raise RuntimeError(f"No se encontraron LAZ tiles en {laz_dir}")

    cloud = load_laz_as_points(laz_tiles, x_min, y_min, x_max, y_max,
                               max_points=points_sample * 2, filter_ground=filter_ground)
    if len(cloud) == 0:
        raise RuntimeError("No se pudieron cargar puntos LiDAR")

    if len(cloud) > points_sample:
        rng = np.random.default_rng(42)
        cloud = cloud[rng.choice(len(cloud), size=points_sample, replace=False)]

    avg_z = float(np.mean(cloud[:, 2]))

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
        mode='markers',
        marker=dict(size=1.8, color=cloud[:, 2], colorscale='Viridis', opacity=0.55,
                    showscale=True, colorbar=dict(title="ElevaciÃ³n (m)")),
        name="LiDAR"
    ))

    for seg in segments_data:
        z = np.full_like(seg["x"], avg_z)
        
        # Build hover text with terrain features if available
        hover_text = []
        for i, si_val in enumerate(seg["si"]):
            text_parts = [f"<b>SI</b>: {si_val:.3f}"]
            
            # Add terrain features to hover text if available
            if seg.get("available_features"):
                for feat in seg["available_features"]:
                    if feat in seg:
                        val = seg[feat][i] if isinstance(seg[feat], np.ndarray) else seg[feat]
                        if pd.notna(val):
                            if feat == 'phi_lidar_deg':
                                text_parts.append(f"<b>Ï† (Â°)</b>: {val:.2f}")
                            elif feat == 'phi_lidar':
                                text_parts.append(f"<b>Ï† (rad)</b>: {val:.4f}")
                            elif feat == 'tri':
                                text_parts.append(f"<b>TRI</b>: {val:.4f}")
                            elif feat == 'ruggedness':
                                text_parts.append(f"<b>Ruggedness</b>: {val:.4f}")
                            elif feat in ['z_min', 'z_max', 'z_mean', 'z_std', 'z_range']:
                                short_name = feat.replace('z_', 'Z_').upper()
                                text_parts.append(f"<b>{short_name}</b>: {val:.2f}m")
            
            hover_text.append("<br>".join(text_parts))
        
        fig.add_trace(go.Scatter3d(
            x=seg["x"], y=seg["y"], z=z,
            mode='lines+markers',
            line=dict(width=5, color='rgba(20,20,20,0.45)'),
            marker=dict(size=3, color=seg["si"], colorscale='RdYlGn', cmin=0, cmax=1,
                        showscale=False),
            name=seg["name"],
            text=hover_text,
            hovertemplate="<b>%{fullData.name}</b><br>X=%{x:.1f}<br>Y=%{y:.1f}<br>%{text}<extra></extra>"
        ))

    first = segments_data[0]
    last = segments_data[-1]
    fig.add_trace(go.Scatter3d(
        x=[first["x"][0]], y=[first["y"][0]], z=[avg_z],
        mode='markers', marker=dict(size=8, color='green'), name='Inicio'))
    fig.add_trace(go.Scatter3d(
        x=[last["x"][-1]], y=[last["y"][-1]], z=[avg_z],
        mode='markers', marker=dict(size=8, color='red'), name='Fin'))

    fig.update_layout(
        title=f"VisualizaciÃ³n 3D | {base_name}",
        scene=dict(
            xaxis_title="UTM X (m)",
            yaxis_title="UTM Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1)),
        ),
        legend=dict(x=0.01, y=0.99),
        width=1400,
        height=900,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out)
        print(f"âœ… VisualizaciÃ³n guardada en: {out}")
    else:
        fig.show()



def main():
    parser = argparse.ArgumentParser(description="VisualizaciÃ³n 3D interactiva de rutas")
    parser.add_argument("--base", required=True)
    parser.add_argument("--mapmatch-dir", default=None)
    parser.add_argument("--laz-dir", default=None)
    parser.add_argument("--points-sample", type=int, default=50_000)
    parser.add_argument("--stability-col", default="si")
    parser.add_argument("--no-ground-filter", action="store_true")
    parser.add_argument("--padding", type=float, default=100.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-terrain-features", action="store_true", default=True,
                       help="Display terrain features in hover tooltips if available")
    args = parser.parse_args()

    try:
        visualize_3d_interactive(
            base_name=args.base,
            mapmatch_dir=args.mapmatch_dir,
            laz_dir=args.laz_dir,
            points_sample=args.points_sample,
            stability_col=args.stability_col,
            output_path=args.output,
            filter_ground=not args.no_ground_filter,
            padding_m=args.padding,
            show_terrain_features=args.show_terrain_features,
        )
    except Exception as exc:
        print(f"âŒ ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


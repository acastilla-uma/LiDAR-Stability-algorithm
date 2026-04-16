#!/usr/bin/env python3
"""
Visualización 3D interactiva de segmentos sobre nube LiDAR usando PyVista.
Genera HTML sin necesidad de X11 (funciona en servidores sin GUI).
Soporta hasta 1M de puntos con renderizado eficiente.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def visualize_3d_pyvista(base_name: str, output_path: str = None, mapmatch_dir: str = None, 
                        laz_dir: str = None, points_sample: int = 1_000_000, 
                        stability_col: str = "si", filter_ground: bool = True, 
                        padding_m: float = 100.0):
    """
    Visualiza segmentos de ruta con nube LiDAR usando PyVista.
    Genera HTML interactivo sin requerir X11.
    
    Args:
        base_name: Nombre base del segmento (ej: DOBACK024_20250929)
        output_path: Ruta del archivo HTML de salida
        mapmatch_dir: Directorio con archivos map-matched
        laz_dir: Directorio con archivos LAZ
        points_sample: Máximo número de puntos a visualizar
        stability_col: Columna de estabilidad
        filter_ground: Filtrar puntos de suelo
        padding_m: Margen alrededor de la ruta en metros
    """
    try:
        import pyvista as pv
        # Desabilitar renderizado en pantalla
        pv.start_xvfb()
    except ImportError:
        logger.error("PyVista no está instalado. Instálalo con: pip install pyvista")
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[3]
    if not output_path:
        output_path = project_root / "output" / f"visualization_3d_{base_name}.html"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        
        seg_data = {"name": seg_file.stem, "x": x, "y": y, "si": si}
        segments_data.append(seg_data)
        all_x.extend(x)
        all_y.extend(y)

    if not segments_data:
        raise RuntimeError("No hay segmentos válidos para visualizar")

    all_x_arr = np.array(all_x)
    all_y_arr = np.array(all_y)

    x_min = all_x_arr.min() - padding_m
    x_max = all_x_arr.max() + padding_m
    y_min = all_y_arr.min() - padding_m
    y_max = all_y_arr.max() + padding_m

    laz_tiles = find_laz_tiles(x_min, y_min, x_max, y_max, laz_dir)
    if not laz_tiles:
        raise FileNotFoundError(f"No se encontraron archivos LAZ en {laz_dir} para las coordenadas especificadas")

    logger.info(f"Encontrados {len(laz_tiles)} archivos LAZ")

    # Cargar nube LiDAR
    cloud = load_laz_as_points(laz_tiles, x_min, y_min, x_max, y_max,
                               max_points=points_sample * 2, 
                               filter_ground=filter_ground)
    
    if len(cloud) > points_sample:
        logger.info(f"Muestreando {points_sample} puntos de {len(cloud)} disponibles")
        rng = np.random.RandomState(42)
        cloud = cloud[rng.choice(len(cloud), size=points_sample, replace=False)]
    
    logger.info(f"Total de puntos LiDAR: {len(cloud)}")

    # Crear geometría con PyVista
    pv.set_jupyter_backend(None)
    plotter = pv.Plotter(off_screen=True)
    
    # Crear punto cloud
    points = cloud[:, :3]
    z_values = cloud[:, 2]
    
    # Crear malla de puntos
    cloud_mesh = pv.PolyData(points)
    
    # Colorear por altura
    z_min, z_max = z_values.min(), z_values.max()
    cloud_mesh["elevation"] = z_values - z_min
    
    # Añadir nube LiDAR con escala de colores
    plotter.add_mesh(
        cloud_mesh,
        scalars="elevation",
        cmap="turbo",
        point_size=2,
        render_points_as_spheres=False,
        opacity=0.7
    )
    
    # Colores para segmentos
    segment_colors = ["red", "green", "blue", "yellow"]
    
    for idx, seg in enumerate(segments_data):
        x, y = seg["x"], seg["y"]
        z = np.ones_like(x) * 640  # Altura aproximada
        
        # Crear línea de ruta
        route_points = np.column_stack([x, y, z])
        route_line = pv.Spline(route_points, n_points=len(route_points) * 2)
        
        color = segment_colors[idx % len(segment_colors)]
        plotter.add_mesh(
            route_line,
            color=color,
            line_width=5,
            label=seg['name']
        )
        
        logger.info(f"  Segmento '{seg['name']}': {len(x)} puntos de ruta")
    
    # Configurar vista
    plotter.camera_position = [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2 + 500
    ]
    plotter.camera.focus = [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        650
    ]
    
    # Mostrar leyenda
    plotter.add_legend()
    
    # Exportar a HTML
    logger.info(f"\nGenerando HTML interactivo en: {output_path}")
    plotter.export_html(str(output_path), iframe=True)
    
    logger.info(f"✅ Visualización guardada en: {output_path}")
    logger.info(f"Abre el archivo en un navegador web para ver la visualización interactiva")
    
    plotter.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualización 3D de rutas con PyVista (genera HTML interactivo)"
    )
    parser.add_argument("--base", required=True, 
                        help="Nombre base del segmento (ej: DOBACK024_20250929)")
    parser.add_argument("--output", default=None,
                        help="Ruta del archivo HTML de salida")
    parser.add_argument("--mapmatch-dir", default=None,
                        help="Directorio con archivos map-matched")
    parser.add_argument("--laz-dir", default=None,
                        help="Directorio con archivos LAZ")
    parser.add_argument("--points-sample", type=int, default=1_000_000,
                        help="Máximo número de puntos a visualizar")
    parser.add_argument("--stability-col", default="si",
                        help="Nombre de la columna de estabilidad")
    parser.add_argument("--no-ground-filter", action="store_true",
                        help="Incluir todos los puntos sin filtrar el suelo")
    parser.add_argument("--padding", type=float, default=100.0,
                        help="Margen alrededor de la ruta en metros")
    
    args = parser.parse_args()

    try:
        filter_ground = not args.no_ground_filter
        visualize_3d_pyvista(
            base_name=args.base,
            output_path=args.output,
            mapmatch_dir=args.mapmatch_dir,
            laz_dir=args.laz_dir,
            points_sample=args.points_sample,
            stability_col=args.stability_col,
            filter_ground=filter_ground,
            padding_m=args.padding
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

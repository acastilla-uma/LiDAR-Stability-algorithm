#!/usr/bin/env python3
"""
Visualización 3D local de segmentos sobre nube LiDAR usando Open3D.
Renderizado potente para millones de puntos con aceleración GPU.
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


def visualize_3d_open3d(base_name: str, mapmatch_dir: str = None, laz_dir: str = None,
                        points_sample: int = 1_000_000, stability_col: str = "si",
                        filter_ground: bool = True, padding_m: float = 100.0, radius_m: float = 5.0):
    """
    Visualiza segmentos de ruta con nube LiDAR usando Open3D.
    
    Args:
        base_name: Nombre base del segmento (ej: DOBACK024_20250929)
        mapmatch_dir: Directorio con archivos map-matched
        laz_dir: Directorio con archivos LAZ
        points_sample: Máximo número de puntos a visualizar
        stability_col: Columna de estabilidad
        filter_ground: Filtrar puntos de suelo
        padding_m: Margen alrededor de la ruta en metros
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("Open3D no está instalado. Instálalo con: pip install open3d")
        logger.info("\nSi tienes problemas de dependencias, usa:")
        logger.info("  conda install -c open3d-admin open3d")
        logger.info("o en Linux:")
        logger.info("  sudo apt install libopengl0 libopengl-dev")
        sys.exit(1)

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
    
    # FILTRADO POR RADIO: Seleccionar solo puntos dentro de 5m de la ruta
    # Construir array de puntos de ruta (XY solamente)
    route_points_xy = []
    for seg in segments_data:
        x, y = seg["x"], seg["y"]
        for xi, yi in zip(x, y):
            route_points_xy.append([xi, yi])
    
    route_points_xy = np.array(route_points_xy)
    radius = radius_m  # metros (parámetro configurable)
    
    # Usar scipy.spatial.distance para calcular distancias mínimas
    from scipy.spatial.distance import cdist
    
    # Calcular distancias XY desde cada punto LiDAR al punto de ruta más cercano
    lidar_xy = cloud[:, :2]  # Solo X, Y
    distances = cdist(lidar_xy, route_points_xy, metric='euclidean')
    min_distances = distances.min(axis=1)  # Distancia mínima a cualquier punto de ruta
    
    # Filtrar: mantener solo puntos dentro del radio
    mask = min_distances <= radius
    cloud = cloud[mask]
    
    logger.info(f"Filtrados puntos dentro de {radius}m: {len(cloud)} de {mask.shape[0]} puntos")
    
    if len(cloud) == 0:
        logger.warning("⚠️  No hay puntos LiDAR dentro del radio de 5m de la ruta")
        logger.info("Intenta aumentar el radio con: --radius 10")
    
    if len(cloud) > points_sample:
        logger.info(f"Muestreando {points_sample} puntos de {len(cloud)} disponibles")
        rng = np.random.RandomState(42)
        cloud = cloud[rng.choice(len(cloud), size=points_sample, replace=False)]
    
    logger.info(f"Total de puntos LiDAR a visualizar: {len(cloud)}")

    # Crear nube de puntos Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    
    # Colorear por altura (eje Z)
    z_values = cloud[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
    
    # Colormap: azul (bajo) -> verde -> amarillo -> rojo (alto)
    colors = np.zeros((len(cloud), 3))
    colors[:, 0] = np.clip(z_normalized * 2 - 1, 0, 1)  # Rojo
    colors[:, 1] = np.clip(1 - np.abs(z_normalized * 2 - 1), 0, 1)  # Verde
    colors[:, 2] = np.clip(1 - z_normalized * 2, 0, 1)  # Azul
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Crear geometrías para los segmentos de ruta
    geometries = [pcd]
    
    # Añadir líneas de ruta para cada segmento
    segment_colors = [
        [1, 0, 0],  # Rojo
        [0, 1, 0],  # Verde
        [0, 0, 1],  # Azul
        [1, 1, 0],  # Amarillo
    ]
    
    for idx, seg in enumerate(segments_data):
        x, y = seg["x"], seg["y"]
        z = np.ones_like(x) * 640  # Altura aproximada de la carretera
        
        # Crear línea de ruta
        points = np.column_stack([x, y, z])
        lines = np.array([[i, i+1] for i in range(len(points)-1)])
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        color = segment_colors[idx % len(segment_colors)]
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(line_set)
        
        logger.info(f"  Segmento '{seg['name']}': {len(x)} puntos de ruta")

    # Crear visualizador
    logger.info("\nAbriendo visualizador 3D...")
    logger.info("Controles:")
    logger.info("  - Botón izquierdo: Rotar")
    logger.info("  - Botón derecho: Trasladar")
    logger.info("  - Rueda: Zoom")
    logger.info("  - 'Z': Resetear vista")
    logger.info("  - 'H': Mostrar ayuda")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Visualización 3D | {base_name}", 
                      width=1280, height=960)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Configurar perspectiva
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, 1])
    view_ctl.set_lookat([
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        650
    ])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.8)
    
    # Ejecutar visualización
    vis.run()
    vis.destroy_window()
    
    logger.info("\nVisualizador cerrado.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualización 3D de rutas con Open3D (GPU-acelerado)"
    )
    parser.add_argument("--base", required=True, 
                        help="Nombre base del segmento (ej: DOBACK024_20250929)")
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
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Radio en metros alrededor de la ruta para filtrar puntos LiDAR")
    
    args = parser.parse_args()

    try:
        filter_ground = not args.no_ground_filter
        visualize_3d_open3d(
            base_name=args.base,
            mapmatch_dir=args.mapmatch_dir,
            laz_dir=args.laz_dir,
            points_sample=args.points_sample,
            stability_col=args.stability_col,
            filter_ground=filter_ground,
            padding_m=args.padding,
            radius_m=args.radius
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

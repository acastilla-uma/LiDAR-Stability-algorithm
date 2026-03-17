#!/usr/bin/env python3
"""
Compute Terrain Features for Route Points

Enriquece datos de ruta map-matched con características de terreno calculadas
desde datos LiDAR (phi_lidar, TRI, ruggedness, etc.).

Columnas nuevas:
  - phi_lidar: Transverse topographic slope (rad y grados)
  - tri: Terrain Roughness Index (m)
  - ruggedness: Terrain ruggedness metric (m)
  - z_min, z_max, z_mean, z_std, z_range: Elevation stats (m)

Uso:
    python Scripts/lidar/compute_route_terrain_features.py \
        --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
        --laz-dir LiDAR-Maps/cnig \
        --output Doback-Data/featured/DOBACK024_20251009_seg87.csv \
        --patch-size 256 \
        --search-radius 100
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from lidar.laz_reader import LAZReader
from lidar.terrain_features import TerrainFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_relevant_laz_tiles(x_center: float, y_center: float, search_radius: float, laz_dir: Path) -> list[Path]:
    """Find LAZ tiles that might contain points near (x, y)."""
    x_min = x_center - search_radius
    x_max = x_center + search_radius
    y_min = y_center - search_radius
    y_max = y_center + search_radius
    
    tx_min = int(x_min // 1000)
    tx_max = int(x_max // 1000)
    ty_min = int(y_min // 1000) + 1
    ty_max = int(y_max // 1000) + 1
    
    tiles = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tiles.extend(laz_dir.glob(f"*_{tx}-{ty}_*.laz"))
            tiles.extend(laz_dir.glob(f"*_{tx}-{ty}.laz"))
    
    return sorted(set(tiles))


def extract_terrain_features_at_point(x: float, y: float, 
                                     laz_tiles: list[Path],
                                     search_radius: float = 100.0,
                                     dem_size: int = 256,
                                     vehicle_track: float = 2.48,
                                     reader_cache: Optional[dict[Path, LAZReader]] = None,
                                     failed_tiles: Optional[set[Path]] = None) -> Dict[str, float]:
    """
    Extract terrain features around a route point.
    
    Args:
        x, y: UTM coordinates
        laz_tiles: List of LAZ file paths
        search_radius: Radius to search for points (m)
        dem_size: Size of DEM grid (dem_size x dem_size)
        vehicle_track: Vehicle track width (m)
        
    Returns:
        Dict with terrain features and NaN if unable to compute
    """
    features = {
        'phi_lidar': np.nan,
        'phi_lidar_deg': np.nan,
        'tri': np.nan,
        'ruggedness': np.nan,
        'z_min': np.nan,
        'z_max': np.nan,
        'z_mean': np.nan,
        'z_std': np.nan,
        'z_range': np.nan,
        'n_points_used': 0,
    }
    
    # Load points from nearby LAZ tiles
    if reader_cache is None:
        reader_cache = {}
    if failed_tiles is None:
        failed_tiles = set()

    all_points = []
    for tile_path in laz_tiles:
        if tile_path in failed_tiles:
            continue

        try:
            reader = reader_cache.get(tile_path)
            if reader is None:
                reader = LAZReader(str(tile_path), filter_ground=False)
                reader_cache[tile_path] = reader

            pts = reader.extract_patch(x_center=x, y_center=y, radius_m=search_radius)
            if pts is not None and len(pts) > 0:
                all_points.append(pts)
        except Exception as e:
            failed_tiles.add(tile_path)
            logger.warning(f"Skipping LAZ tile after read error ({tile_path.name}): {e}")
    
    if not all_points:
        logger.debug(f"No LiDAR points found near ({x:.1f}, {y:.1f})")
        return features
    
    cloud = np.vstack(all_points)
    features['n_points_used'] = len(cloud)
    
    if len(cloud) < 10:
        logger.debug(f"Too few points ({len(cloud)}) near ({x:.1f}, {y:.1f})")
        return features
    
    # Create grid-based DEM
    x_min, x_max = cloud[:, 0].min(), cloud[:, 0].max()
    y_min, y_max = cloud[:, 1].min(), cloud[:, 1].max()
    
    # Expand grid slightly if needed
    if x_max - x_min < 10:
        x_mid = (x_min + x_max) / 2
        x_min, x_max = x_mid - 5, x_mid + 5
    if y_max - y_min < 10:
        y_mid = (y_min + y_max) / 2
        y_min, y_max = y_mid - 5, y_mid + 5
    
    # Create regular grid
    x_grid = np.linspace(x_min, x_max, dem_size)
    y_grid = np.linspace(y_min, y_max, dem_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Interpolate elevation to grid
    try:
        dem = griddata(
            cloud[:, :2],
            cloud[:, 2],
            (xx, yy),
            method='linear'
        ).astype(np.float32)
    except Exception as e:
        logger.debug(f"Grid interpolation failed at ({x:.1f}, {y:.1f}): {e}")
        return features
    
    # Extract features from DEM
    try:
        terrain_features = TerrainFeatureExtractor.extract_features(
            dem, 
            vehicle_track=vehicle_track,
            resolution=1.0
        )
        
        features['phi_lidar'] = terrain_features['phi_lidar']
        features['phi_lidar_deg'] = float(np.degrees(terrain_features['phi_lidar']))
        features['tri'] = terrain_features['tri']
        features['ruggedness'] = terrain_features['ruggedness']
        features['z_min'] = terrain_features['z_min']
        features['z_max'] = terrain_features['z_max']
        features['z_mean'] = terrain_features['z_mean']
        features['z_std'] = terrain_features['z_std']
        features['z_range'] = terrain_features['z_range']
    
    except Exception as e:
        logger.debug(f"Feature extraction failed at ({x:.1f}, {y:.1f}): {e}")
    
    return features


def enrich_route_with_terrain_features(mapmatch_path: str, 
                                      laz_dir: str = None,
                                      output_path: Optional[str] = None,
                                      search_radius: float = 100.0,
                                      dem_size: int = 256,
                                      vehicle_track: float = 2.48,
                                      sampling: int = 1) -> pd.DataFrame:
    """
    Add terrain features to map-matched route data.
    
    Args:
        mapmatch_path: Path to map-matched CSV
        laz_dir: Directory with LAZ files
        output_path: Where to save enriched CSV
        search_radius: Radius to search for LiDAR points around each route point (m)
        dem_size: Size of DEM grid for feature extraction
        vehicle_track: Vehicle track width (m)
        sampling: Process every nth point (1 = all points)
        
    Returns:
        Enriched DataFrame
    """
    mapmatch_path = Path(mapmatch_path)
    if not mapmatch_path.exists():
        raise FileNotFoundError(f"Map-matched file not found: {mapmatch_path}")
    
    project_root = Path(__file__).parent.parent.parent
    laz_dir = Path(laz_dir) if laz_dir else project_root / "LiDAR-Maps" / "cnig"
    
    logger.info(f"Loading map-matched data: {mapmatch_path}")
    df = pd.read_csv(mapmatch_path)
    
    required = {'x_utm', 'y_utm'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    
    # Initialize terrain feature columns
    feature_columns = [
        'phi_lidar', 'phi_lidar_deg', 'tri', 'ruggedness',
        'z_min', 'z_max', 'z_mean', 'z_std', 'z_range', 'n_points_used'
    ]
    for col in feature_columns:
        df[col] = np.nan
    
    # Get unique route points (deduplicate if needed)
    df = df.dropna(subset=['x_utm', 'y_utm'])
    
    # Sample points if requested
    if sampling > 1:
        indices = np.arange(0, len(df), sampling)
    else:
        indices = np.arange(len(df))
    
    logger.info(f"Computing terrain features for {len(indices)} points")
    
    # Reuse readers across points and skip broken tiles after first failure
    reader_cache: dict[Path, LAZReader] = {}
    failed_tiles: set[Path] = set()

    # Process each point
    for i in tqdm(indices, desc="Extracting terrain features"):
        x = df.loc[i, 'x_utm']
        y = df.loc[i, 'y_utm']
        
        # Find relevant LAZ tiles
        laz_tiles = find_relevant_laz_tiles(x, y, search_radius, laz_dir)
        if not laz_tiles:
            logger.debug(f"No LAZ tiles found near ({x:.1f}, {y:.1f})")
            continue
        
        # Extract features
        features = extract_terrain_features_at_point(
            x, y,
            laz_tiles,
            search_radius=search_radius,
            dem_size=dem_size,
            vehicle_track=vehicle_track,
            reader_cache=reader_cache,
            failed_tiles=failed_tiles,
        )
        
        # Store in dataframe
        for col in feature_columns:
            df.loc[i, col] = features[col]
    
    # Interpolate missing values (forward fill then backward fill)
    for col in feature_columns:
        if col != 'n_points_used':
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    logger.info(f"✅ Enrichment complete")
    if failed_tiles:
        logger.warning(f"LAZ tiles skipped due to read errors: {len(failed_tiles)}")
    
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved to: {out}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Enrich map-matched routes with terrain features from LiDAR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mapmatch", required=True, help="Map-matched CSV file")
    parser.add_argument("--laz-dir", default=None, help="Directory with LAZ files")
    parser.add_argument("--output", default=None, help="Output CSV (default: Doback-Data/featured/<name>.csv)")
    parser.add_argument("--search-radius", type=float, default=100.0, 
                       help="Search radius for LiDAR points (m)")
    parser.add_argument("--dem-size", type=int, default=256,
                       help="DEM grid size (pixels per side)")
    parser.add_argument("--vehicle-track", type=float, default=2.48,
                       help="Vehicle track width (m)")
    parser.add_argument("--sampling", type=int, default=1,
                       help="Process every nth point")
    
    args = parser.parse_args()
    
    try:
        mapmatch_path = Path(args.mapmatch)
        if args.output:
            output = args.output
        else:
            project_root = Path(__file__).parent.parent.parent
            output = str(project_root / "Doback-Data" / "featured" / mapmatch_path.name)
        enrich_route_with_terrain_features(
            mapmatch_path=args.mapmatch,
            laz_dir=args.laz_dir,
            output_path=output,
            search_radius=args.search_radius,
            dem_size=args.dem_size,
            vehicle_track=args.vehicle_track,
            sampling=args.sampling
        )
        return 0
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Compute Terrain Features for Route Points

Enriquece datos de ruta map-matched con caracterÃ­sticas de terreno calculadas
desde datos LiDAR (phi_lidar, TRI, ruggedness, etc.).

Columnas nuevas:
  - phi_lidar: Transverse topographic slope (rad y grados)
  - tri: Terrain Roughness Index (m)
  - ruggedness: Terrain ruggedness metric (m)
  - z_min, z_max, z_mean, z_std, z_range: Elevation stats (m)

Uso:
    python src/lidar_stability/lidar/compute_route_terrain_features.py \
        --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
        --laz-dir LiDAR-Maps/cnig \
        --output Doback-Data/featured/DOBACK024_20251009_seg87.csv \
        --patch-size 256 \
        --search-radius 100

    python src/lidar_stability/lidar/compute_route_terrain_features.py \
        --doback DOBACK023 \
        --laz-dir LiDAR-Maps/cnig \
        --search-radius 100
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.lidar.laz_reader import LAZReader
from lidar_stability.lidar.terrain_features import TerrainFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'phi_lidar', 'phi_lidar_deg', 'tri', 'ruggedness',
    'z_min', 'z_max', 'z_mean', 'z_std', 'z_range', 'n_points_used'
]

FINITE_FEATURE_COLUMNS = [
    'phi_lidar', 'phi_lidar_deg', 'tri', 'ruggedness',
    'z_min', 'z_max', 'z_mean', 'z_std', 'z_range'
]

_TILE_COORD_RE = re.compile(r"_(\d+)-(\d+)(?:_|\.laz$)", re.IGNORECASE)


def _are_features_finite(features: Dict[str, float]) -> bool:
    """Return True only if all terrain feature outputs are finite values."""
    return all(np.isfinite(features.get(col, np.nan)) for col in FINITE_FEATURE_COLUMNS)


def _build_laz_tile_index(laz_dir: Path) -> dict[tuple[int, int], list[Path]]:
    """Build an in-memory index from tile integer coordinates to LAZ files."""
    tile_index: dict[tuple[int, int], list[Path]] = {}
    for tile_path in laz_dir.glob("*.laz"):
        match = _TILE_COORD_RE.search(tile_path.name)
        if not match:
            continue
        tx = int(match.group(1))
        ty = int(match.group(2))
        tile_index.setdefault((tx, ty), []).append(tile_path)
    return tile_index


def find_relevant_laz_tiles(
    x_center: float,
    y_center: float,
    search_radius: float,
    laz_dir: Path,
    tile_index: Optional[dict[tuple[int, int], list[Path]]] = None,
) -> list[Path]:
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
            if tile_index is not None:
                tiles.extend(tile_index.get((tx, ty), []))
                continue
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

    if not laz_tiles:
        return features
    
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
    finite_cloud_mask = np.isfinite(cloud).all(axis=1)
    cloud = cloud[finite_cloud_mask]
    features['n_points_used'] = int(len(cloud))
    
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

    if dem is None:
        return features

    dem = np.where(np.isfinite(dem), dem, np.nan)
    if np.count_nonzero(np.isfinite(dem)) < 25:
        logger.debug(f"Insufficient finite DEM cells near ({x:.1f}, {y:.1f})")
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

        if not _are_features_finite(features):
            logger.debug(f"Invalid terrain features (NaN/inf) at ({x:.1f}, {y:.1f})")
            for col in FINITE_FEATURE_COLUMNS:
                features[col] = np.nan
    
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
    
    project_root = Path(__file__).resolve().parents[3]
    laz_dir = Path(laz_dir) if laz_dir else project_root / "LiDAR-Maps" / "cnig"
    
    logger.info(f"Loading map-matched data: {mapmatch_path}")
    df = pd.read_csv(mapmatch_path)
    
    required = {'x_utm', 'y_utm'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    
    # Initialize terrain feature columns
    feature_columns = FEATURE_COLUMNS
    for col in feature_columns:
        df[col] = np.nan

    if not laz_dir.exists() or not laz_dir.is_dir():
        logger.warning(f"LAZ directory not found or invalid: {laz_dir}. Skipping terrain feature computation.")
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            logger.info(f"Saved to: {out}")
        return df

    laz_inventory = next(laz_dir.glob("*.laz"), None)
    if laz_inventory is None:
        logger.warning(f"No LAZ files found in {laz_dir}. Skipping terrain feature computation.")
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            logger.info(f"Saved to: {out}")
        return df
    
    tile_index = _build_laz_tile_index(laz_dir)
    logger.info(f"Indexed {sum(len(v) for v in tile_index.values())} LAZ files into {len(tile_index)} tiles")

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
    tile_query_cache: dict[tuple[int, int, int, int], list[Path]] = {}

    x_values = pd.to_numeric(df['x_utm'], errors='coerce').to_numpy(dtype=float)
    y_values = pd.to_numeric(df['y_utm'], errors='coerce').to_numpy(dtype=float)
    feature_values = {
        col: np.full(len(df), np.nan, dtype=float)
        for col in feature_columns
    }

    # Process each point
    for i in tqdm(indices, desc="Extracting terrain features"):
        x = x_values[i]
        y = y_values[i]

        if not np.isfinite(x) or not np.isfinite(y):
            logger.debug(f"Skipping point with invalid coordinates at index {i}: ({x}, {y})")
            continue
        
        # Find relevant LAZ tiles
        x_min = x - search_radius
        x_max = x + search_radius
        y_min = y - search_radius
        y_max = y + search_radius
        query_key = (
            int(x_min // 1000),
            int(x_max // 1000),
            int(y_min // 1000) + 1,
            int(y_max // 1000) + 1,
        )
        laz_tiles = tile_query_cache.get(query_key)
        if laz_tiles is None:
            laz_tiles = find_relevant_laz_tiles(
                x,
                y,
                search_radius,
                laz_dir,
                tile_index=tile_index,
            )
            tile_query_cache[query_key] = laz_tiles
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
        
        # Store values in numpy arrays and assign to dataframe in one pass.
        for col in feature_columns:
            feature_values[col][i] = features[col]

    for col in feature_columns:
        df[col] = feature_values[col]

    for col in feature_columns:
        if col == 'n_points_used':
            continue
        finite_mask = np.isfinite(df[col])
        df.loc[~finite_mask, col] = np.nan
    
    # Interpolate missing values (forward fill then backward fill)
    for col in feature_columns:
        if col != 'n_points_used':
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    logger.info(f"âœ… Enrichment complete")
    if failed_tiles:
        logger.warning(f"LAZ tiles skipped due to read errors: {len(failed_tiles)}")
    
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved to: {out}")
    
    return df


def normalize_doback_id(raw_doback: str) -> str:
    """Normalize DOBACK identifier to canonical format (e.g., DOBACK023)."""
    token = raw_doback.strip().upper()
    if token.startswith("DOBACK"):
        return token
    if token.isdigit():
        return f"DOBACK{int(token):03d}"
    return token


def enrich_doback_batch(doback: str,
                        mapmatch_dir: Optional[str] = None,
                        featured_dir: Optional[str] = None,
                        laz_dir: Optional[str] = None,
                        search_radius: float = 100.0,
                        dem_size: int = 256,
                        vehicle_track: float = 2.48,
                        sampling: int = 1) -> tuple[int, int]:
    """Process all map-matched CSV files for a given DOBACK ID."""
    project_root = Path(__file__).resolve().parents[3]
    doback_id = normalize_doback_id(doback)
    mapmatch_root = Path(mapmatch_dir) if mapmatch_dir else project_root / "Doback-Data" / "map-matched"
    featured_root = Path(featured_dir) if featured_dir else project_root / "Doback-Data" / "featured"

    files = sorted(mapmatch_root.glob(f"{doback_id}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No map-matched files found for {doback_id} in {mapmatch_root}")

    logger.info(f"Found {len(files)} files for {doback_id}")
    success = 0
    failed = 0

    for mapmatch_file in files:
        output_file = featured_root / mapmatch_file.name
        logger.info(f"Processing: {mapmatch_file.name}")
        try:
            enrich_route_with_terrain_features(
                mapmatch_path=str(mapmatch_file),
                laz_dir=laz_dir,
                output_path=str(output_file),
                search_radius=search_radius,
                dem_size=dem_size,
                vehicle_track=vehicle_track,
                sampling=sampling,
            )
            success += 1
        except Exception as exc:
            failed += 1
            logger.error(f"Failed {mapmatch_file.name}: {exc}")

    logger.info(f"Batch finished for {doback_id}: {success} ok, {failed} failed")
    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Enrich map-matched routes with terrain features from LiDAR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mapmatch", help="Map-matched CSV file (single-file mode)")
    mode_group.add_argument("--doback", help="DOBACK ID to process all its map-matched files (e.g., DOBACK023 or 23)")
    parser.add_argument("--laz-dir", default=None, help="Directory with LAZ files")
    parser.add_argument("--output", default=None, help="Output CSV (single-file mode only)")
    parser.add_argument("--mapmatch-dir", default=None, help="Directory with map-matched CSVs (batch mode)")
    parser.add_argument("--featured-dir", default=None, help="Output directory for enriched CSVs (batch mode)")
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
        if args.mapmatch:
            mapmatch_path = Path(args.mapmatch)
            if args.output:
                output = args.output
            else:
                project_root = Path(__file__).resolve().parents[3]
                output = str(project_root / "Doback-Data" / "featured" / mapmatch_path.name)

            enrich_route_with_terrain_features(
                mapmatch_path=args.mapmatch,
                laz_dir=args.laz_dir,
                output_path=output,
                search_radius=args.search_radius,
                dem_size=args.dem_size,
                vehicle_track=args.vehicle_track,
                sampling=args.sampling,
            )
        else:
            if args.output:
                raise ValueError("--output is only valid with --mapmatch. Use --featured-dir in batch mode.")
            _, failed = enrich_doback_batch(
                doback=args.doback,
                mapmatch_dir=args.mapmatch_dir,
                featured_dir=args.featured_dir,
                laz_dir=args.laz_dir,
                search_radius=args.search_radius,
                dem_size=args.dem_size,
                vehicle_track=args.vehicle_track,
                sampling=args.sampling,
            )
            if failed > 0:
                return 1
        return 0
    except Exception as e:
        logger.error(f"âŒ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


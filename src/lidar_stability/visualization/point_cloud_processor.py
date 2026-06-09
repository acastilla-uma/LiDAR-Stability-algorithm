"""
Point Cloud Processing

Loads LAZ files, extracts points near route, and decimates for visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class PointCloudData:
    """Processed point cloud data for visualization."""
    points_xyz: np.ndarray  # Shape: (N, 3) - X, Y, Z coordinates
    elevations: np.ndarray  # Shape: (N,) - Z values
    points_count_raw: int  # Count before decimation
    points_count_decimated: int  # Count after decimation
    decimation_ratio: float
    
    @property
    def num_points(self) -> int:
        """Number of decimated points for visualization."""
        return len(self.points_xyz)


def _load_laz_file(laz_path: Path, use_all_points: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load LAZ file and extract ground points (ASPRS class 2).
    
    Args:
        laz_path: Path to LAZ file
        
    Returns:
        Tuple of (points_xyz, elevations) where points_xyz is (N, 3) array
        
    Raises:
        ImportError: If laspy not available
    """
    if not HAS_LASPY:
        raise ImportError("laspy required for LAZ processing. Install with: pip install laspy[lazrs]")
    
    logger.info(f"Loading LAZ file: {laz_path.name}")
    
    las = laspy.read(str(laz_path))
    logger.info(f"  Total points in file: {len(las)}")
    
    # Filter for ground points (ASPRS classification = 2) unless use_all_points is True
    if use_all_points:
        ground_points = las
        logger.info(f"  Using all points from file: {len(ground_points)}")
    else:
        if hasattr(las, "classification"):
            ground_mask = las.classification == 2
            ground_points = las[ground_mask]
            logger.info(f"  Ground points (class 2): {len(ground_points)}")
        else:
            ground_points = las
            logger.warning("  No classification field found, using all points")
    
    # Extract XYZ coordinates
    xyz = np.vstack([ground_points.x, ground_points.y, ground_points.z]).T
    elevations = ground_points.z
    
    return xyz, elevations


def _get_laz_bounds(laz_path: Path) -> tuple[float, float, float, float]:
    """Return the (xmin, xmax, ymin, ymax) bounds from the LAZ header."""
    if not HAS_LASPY:
        raise ImportError("laspy required for LAZ processing. Install with: pip install laspy[lazrs]")

    with laspy.open(str(laz_path)) as reader:
        header = reader.header
        return (
            float(header.min[0]),
            float(header.max[0]),
            float(header.min[1]),
            float(header.max[1]),
        )


def _boxes_overlap(
    a_min_x: float,
    a_max_x: float,
    a_min_y: float,
    a_max_y: float,
    b_min_x: float,
    b_max_x: float,
    b_min_y: float,
    b_max_y: float,
) -> bool:
    return not (
        a_max_x < b_min_x
        or a_min_x > b_max_x
        or a_max_y < b_min_y
        or a_min_y > b_max_y
    )


def _filter_points_by_route(
    xyz: np.ndarray,
    route_x_utm: np.ndarray,
    route_y_utm: np.ndarray,
    buffer_radius: float,
) -> np.ndarray:
    """
    Filter point cloud to keep only points within buffer_radius of route.
    
    Args:
        xyz: Point cloud coordinates (N, 3)
        route_x_utm, route_y_utm: Route coordinates in UTM (M,)
        buffer_radius: Buffer radius in meters
        
    Returns:
        Boolean mask of points within buffer
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)

    # Build KDTree of route points (only x, y)
    route_xy = np.vstack([route_x_utm, route_y_utm]).T
    tree = cKDTree(route_xy)
    
    # Query distances from each point to nearest route point
    distances, _ = tree.query(xyz[:, :2])
    
    # Keep points within buffer
    mask = distances <= buffer_radius
    logger.info(
        f"Filtered to {mask.sum()} points within {buffer_radius}m buffer "
        f"({100 * mask.sum() / len(xyz):.1f}% of total)"
    )
    
    return mask


def _decimate_points(
    xyz: np.ndarray,
    decimation_ratio: float,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    Randomly sample points for visualization.
    
    Args:
        xyz: Point cloud coordinates (N, 3)
        decimation_ratio: Fraction of points to keep (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (decimated_xyz, decimation_count)
    """
    if not 0 < decimation_ratio <= 1:
        raise ValueError(f"decimation_ratio must be in (0, 1], got {decimation_ratio}")
    
    n_original = len(xyz)
    n_keep = max(1, int(n_original * decimation_ratio))
    
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_original, size=n_keep, replace=False)
    
    decimated = xyz[indices]
    logger.info(f"Decimated {n_original} → {len(decimated)} points "
               f"({decimation_ratio * 100:.1f}%)")
    
    return decimated, n_original


def process_segment_point_cloud(
    segment_data,  # SegmentData object
    buffer_radius: float = 5.0,
    decimation_ratio: float = 0.1,
    use_all_points: bool = False,
) -> PointCloudData:
    """
    Load, filter, and decimate point cloud for segment visualization.
    
    Args:
        segment_data: SegmentData object with segment info and LAZ tile list
        buffer_radius: Buffer radius in meters around route
        decimation_ratio: Fraction of points to keep after filtering (0.0-1.0)
        
    Returns:
        PointCloudData with processed points
        
    Raises:
        ValueError: If no LAZ tiles found or no points extracted
    """
    if not segment_data.laz_tiles:
        raise ValueError(f"No LAZ tiles found for segment {segment_data.segment_id}")
    
    logger.info(f"Processing point cloud for {segment_data.segment_id}")
    logger.info(f"  Buffer radius: {buffer_radius}m")
    logger.info(f"  Decimation ratio: {decimation_ratio}")

    route_x_utm = segment_data.df["x_utm"].values
    route_y_utm = segment_data.df["y_utm"].values
    route_min_x = float(route_x_utm.min())
    route_max_x = float(route_x_utm.max())
    route_min_y = float(route_y_utm.min())
    route_max_y = float(route_y_utm.max())
    buffered_min_x = route_min_x - buffer_radius
    buffered_max_x = route_max_x + buffer_radius
    buffered_min_y = route_min_y - buffer_radius
    buffered_max_y = route_max_y + buffer_radius

    logger.info(
        "  Route bounds X=[%.2f, %.2f], Y=[%.2f, %.2f]",
        route_min_x,
        route_max_x,
        route_min_y,
        route_max_y,
    )
    
    # Load all LAZ tiles
    all_xyz = []
    overlapping_tiles = []
    for laz_path in segment_data.laz_tiles:
        try:
            tile_min_x, tile_max_x, tile_min_y, tile_max_y = _get_laz_bounds(laz_path)
            if not _boxes_overlap(
                buffered_min_x,
                buffered_max_x,
                buffered_min_y,
                buffered_max_y,
                tile_min_x,
                tile_max_x,
                tile_min_y,
                tile_max_y,
            ):
                logger.info(
                    "Skipping %s: tile bounds X=[%.2f, %.2f], Y=[%.2f, %.2f] do not overlap buffered route",
                    laz_path.name,
                    tile_min_x,
                    tile_max_x,
                    tile_min_y,
                    tile_max_y,
                )
                continue

            overlapping_tiles.append(laz_path)
            xyz, _ = _load_laz_file(laz_path, use_all_points=use_all_points)
            all_xyz.append(xyz)
        except Exception as e:
            logger.warning(f"Error loading {laz_path.name}: {e}")
    
    if not overlapping_tiles:
        tile_summaries = []
        for laz_path in segment_data.laz_tiles[:8]:
            try:
                tile_min_x, tile_max_x, tile_min_y, tile_max_y = _get_laz_bounds(laz_path)
                tile_summaries.append(
                    f"{laz_path.name} X=[{tile_min_x:.2f}, {tile_max_x:.2f}], Y=[{tile_min_y:.2f}, {tile_max_y:.2f}]"
                )
            except Exception:
                tile_summaries.append(laz_path.name)

        raise ValueError(
            f"No LAZ tiles overlap the buffered route for {segment_data.segment_id}. "
            f"Buffered route X=[{buffered_min_x:.2f}, {buffered_max_x:.2f}], "
            f"Y=[{buffered_min_y:.2f}, {buffered_max_y:.2f}]. "
            f"Available tiles: {', '.join(tile_summaries)}"
        )

    if not all_xyz:
        raise ValueError(f"Could not load any overlapping LAZ tiles for {segment_data.segment_id}")
    
    # Combine all points
    combined_xyz = np.vstack(all_xyz)
    logger.info(f"Combined {len(combined_xyz)} points from {len(all_xyz)} LAZ files")

    if len(combined_xyz) == 0:
        raise ValueError(
            f"No ground points (class 2) were extracted from the selected LAZ tiles for {segment_data.segment_id}. "
            "This usually means the chosen source tiles have no class 2 points or the source is not a ground-classified dataset."
        )
    
    buffer_mask = _filter_points_by_route(
        combined_xyz, route_x_utm, route_y_utm, buffer_radius
    )
    buffered_xyz = combined_xyz[buffer_mask]
    
    if len(buffered_xyz) == 0:
        tile_names = ", ".join(p.name for p in segment_data.laz_tiles[:8])
        if len(segment_data.laz_tiles) > 8:
            tile_names += f" ... (+{len(segment_data.laz_tiles) - 8} more)"
        raise ValueError(
            f"No points found within {buffer_radius}m buffer of route. "
            f"Route bounds X=[{route_x_utm.min():.2f}, {route_x_utm.max():.2f}], "
            f"Y=[{route_y_utm.min():.2f}, {route_y_utm.max():.2f}]. "
            f"Loaded LAZ tiles: {tile_names}. "
            "This usually means the selected route slice does not overlap the chosen LAZ coverage. "
            "Try a wider point range, remove point slicing, or use a different LAZ source."
        )
    
    # Decimate
    decimated_xyz, n_raw = _decimate_points(buffered_xyz, decimation_ratio)
    
    return PointCloudData(
        points_xyz=decimated_xyz,
        elevations=decimated_xyz[:, 2],
        points_count_raw=len(buffered_xyz),
        points_count_decimated=len(decimated_xyz),
        decimation_ratio=decimation_ratio,
    )

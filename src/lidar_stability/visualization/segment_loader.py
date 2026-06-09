"""
Segment Data Loader

Loads featured CSV data for a segment and identifies relevant LAZ tiles.
Handles geographic bounds extraction and coordinate system conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Iterable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentData:
    """Loaded segment data and metadata."""
    segment_id: str
    csv_path: Path
    df: pd.DataFrame
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    x_utm_min: float
    x_utm_max: float
    y_utm_min: float
    y_utm_max: float
    laz_dir: Union[Path, list[Path]]
    laz_tiles: list[Path]
    
    @property
    def num_points(self) -> int:
        """Total number of measurement points in segment."""
        return len(self.df)
    
    @property
    def center_lat(self) -> float:
        """Center latitude of segment bounds."""
        return (self.lat_min + self.lat_max) / 2
    
    @property
    def center_lon(self) -> float:
        """Center longitude of segment bounds."""
        return (self.lon_min + self.lon_max) / 2


def _find_laz_tiles(
    x_utm_min: float,
    x_utm_max: float,
    y_utm_min: float,
    y_utm_max: float,
    laz_dir: Union[Path, Iterable[Path]],
    search_buffer: int = 1000,
) -> list[Path]:
    """
    Find LAZ tiles that cover geographic bounds.
    
    LAZ tiles are organized in 1000m × 1000m grid with naming pattern:
    *_<tx>-<ty>.laz  where tx = floor(x_utm / 1000), ty = floor(y_utm / 1000)
    
    Args:
        x_utm_min, x_utm_max: UTM X bounds (meters)
        y_utm_min, y_utm_max: UTM Y bounds (meters)
        laz_dir: Path to LAZ directory (e.g., LiDAR-Maps/cnig/)
        search_buffer: Additional tiles to search beyond bounds (meters)
        
    Returns:
        List of found LAZ tile paths
    """
    # Normalize laz_dir to iterable
    if isinstance(laz_dir, Path):
        laz_dirs = [laz_dir]
    else:
        laz_dirs = list(laz_dir)

    # Add search buffer
    x_min_buffered = x_utm_min - search_buffer
    x_max_buffered = x_utm_max + search_buffer
    y_min_buffered = y_utm_min - search_buffer
    y_max_buffered = y_utm_max + search_buffer
    
    # Calculate tile indices
    tx_min = int(x_min_buffered // 1000)
    tx_max = int(x_max_buffered // 1000) + 1
    ty_min = int(y_min_buffered // 1000)
    ty_max = int(y_max_buffered // 1000) + 1
    
    logger.info(f"LAZ tile search: tx=[{tx_min}, {tx_max}], ty=[{ty_min}, {ty_max}]")
    logger.info(f"  Searching LAZ dirs: {', '.join(str(p) for p in laz_dirs)}")
    
    # Build search patterns. Geo-mad tiles are often named like "428-4481.laz".
    # CNIG tiles often include a prefix such as "PNOA_2024_MAD_428-4481_NPC01.laz".
    found_tiles = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            for d in laz_dirs:
                matches: list[Path] = []
                for pattern in (
                    f"*{tx}-{ty}.laz",
                    f"*{tx}-{ty}_*.laz",
                    f"*_{tx}-{ty}*.laz",
                ):
                    try:
                        matches.extend(Path(d).glob(pattern))
                    except Exception:
                        continue

                if matches:
                    # Preserve first-found order but avoid duplicates.
                    unique_matches = []
                    seen = set()
                    for match in matches:
                        if match not in seen:
                            unique_matches.append(match)
                            seen.add(match)
                    found_tiles.extend(unique_matches)
                    logger.info(f"  Found tile {tx}-{ty} in {d}: {unique_matches[0].name}")
    
    if not found_tiles:
        logger.warning(f"No LAZ tiles found for bounds x=[{x_utm_min}, {x_utm_max}], "
                      f"y=[{y_utm_min}, {y_utm_max}]")
    
    return found_tiles


def load_segment_data(
    segment_id: Union[str, Path],
    featured_data_dir: Path,
    laz_dir: Union[Path, Iterable[Path]],
) -> SegmentData:
    """
    Load featured CSV for a segment and find associated LAZ tiles.
    
    Args:
        segment_id: Segment identifier (e.g., DOBACK024_20251007_seg28)
        featured_data_dir: Path to featured data directory (Doback-Data/featured/)
        laz_dir: Path to LAZ directory (LiDAR-Maps/cnig/)
        
    Returns:
        SegmentData object with CSV loaded and LAZ tiles identified
        
    Raises:
        FileNotFoundError: If segment CSV not found
    """
    # Allow passing either a segment ID or a full path to a CSV file.
    if isinstance(segment_id, (str, Path)) and Path(segment_id).exists():
        # User provided a path to a CSV file
        csv_path = Path(segment_id)
        segment_id_str = csv_path.stem
    else:
        segment_id_str = str(segment_id)
        csv_path = featured_data_dir / f"{segment_id_str}.csv"
        # If not found in featured, try filtered_featured_geomad sibling folder
        if not csv_path.exists():
            alt_dir = featured_data_dir.parent / "filtered_featured_geomad"
            alt_path = alt_dir / f"{segment_id_str}.csv"
            if alt_path.exists():
                csv_path = alt_path
            else:
                raise FileNotFoundError(f"Segment CSV not found: {csv_path}")
    
    logger.info(f"Loading segment data: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Verify required columns
    required_cols = ["lat", "lon", "x_utm", "y_utm", "si"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in segment CSV: {missing}")
    
    # Extract geographic bounds
    lat_min = df["lat"].min()
    lat_max = df["lat"].max()
    lon_min = df["lon"].min()
    lon_max = df["lon"].max()
    x_utm_min = df["x_utm"].min()
    x_utm_max = df["x_utm"].max()
    y_utm_min = df["y_utm"].min()
    y_utm_max = df["y_utm"].max()
    
    logger.info(f"  Geographic bounds:")
    logger.info(f"    Lat: [{lat_min:.6f}, {lat_max:.6f}]")
    logger.info(f"    Lon: [{lon_min:.6f}, {lon_max:.6f}]")
    logger.info(f"    UTM X: [{x_utm_min:.1f}, {x_utm_max:.1f}]")
    logger.info(f"    UTM Y: [{y_utm_min:.1f}, {y_utm_max:.1f}]")
    
    # Find LAZ tiles
    laz_tiles = _find_laz_tiles(x_utm_min, x_utm_max, y_utm_min, y_utm_max, laz_dir)
    
    return SegmentData(
        segment_id=segment_id_str,
        csv_path=csv_path,
        df=df,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        x_utm_min=x_utm_min,
        x_utm_max=x_utm_max,
        y_utm_min=y_utm_min,
        y_utm_max=y_utm_max,
        laz_dir=laz_dir,
        laz_tiles=laz_tiles,
    )


def slice_segment_data(
    segment_data: SegmentData,
    point_start: int = 0,
    point_end: Optional[int] = None,
    point_step: int = 1,
) -> SegmentData:
    """
    Select a subset of route points from a segment.

    Args:
        segment_data: Original segment data
        point_start: Start index (inclusive)
        point_end: End index (exclusive), None means until end
        point_step: Sampling step for points (every Nth point)

    Returns:
        New SegmentData with selected route points and recomputed bounds/tiles

    Raises:
        ValueError: If selection yields fewer than 2 points
    """
    if point_start < 0:
        raise ValueError("point_start must be >= 0")
    if point_step <= 0:
        raise ValueError("point_step must be > 0")

    df = segment_data.df.sort_values("timestamp")
    selected_df = df.iloc[point_start:point_end:point_step].copy()

    if len(selected_df) < 2:
        raise ValueError(
            "Selected route must contain at least 2 points. "
            f"Got {len(selected_df)} with start={point_start}, end={point_end}, step={point_step}."
        )

    lat_min = selected_df["lat"].min()
    lat_max = selected_df["lat"].max()
    lon_min = selected_df["lon"].min()
    lon_max = selected_df["lon"].max()
    x_utm_min = selected_df["x_utm"].min()
    x_utm_max = selected_df["x_utm"].max()
    y_utm_min = selected_df["y_utm"].min()
    y_utm_max = selected_df["y_utm"].max()

    laz_tiles = _find_laz_tiles(
        x_utm_min=x_utm_min,
        x_utm_max=x_utm_max,
        y_utm_min=y_utm_min,
        y_utm_max=y_utm_max,
        laz_dir=segment_data.laz_dir,
    )

    logger.info(
        "Selected route points: %d -> %d (start=%d, end=%s, step=%d)",
        len(segment_data.df),
        len(selected_df),
        point_start,
        "None" if point_end is None else point_end,
        point_step,
    )

    return SegmentData(
        segment_id=segment_data.segment_id,
        csv_path=segment_data.csv_path,
        df=selected_df,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        x_utm_min=x_utm_min,
        x_utm_max=x_utm_max,
        y_utm_min=y_utm_min,
        y_utm_max=y_utm_max,
        laz_dir=segment_data.laz_dir,
        laz_tiles=laz_tiles,
    )

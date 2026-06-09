#!/usr/bin/env python3
"""
Visualize a single segment non-interactively.

Searches LAZ tiles first in `LiDAR-Maps/geo-mad` then in `LiDAR-Maps/cnig` (or provided --laz-dir).

Usage:
  python -m src.lidar_stability.visualization.visualize_segment \
    SEGMENT_ID --buffer-radius 5 --decimation-ratio 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = SRC_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.visualization.segment_loader import (
    load_segment_data,
    slice_segment_data,
)
from lidar_stability.visualization.segment_ranking import SegmentScore
from lidar_stability.visualization.point_cloud_processor import (
    process_segment_point_cloud,
)
from lidar_stability.visualization.map_builder import create_segment_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize a single segment")
    parser.add_argument("segment_id", type=str, help="Segment identifier (e.g., DOBACK024_20251007_seg28)")
    parser.add_argument("--buffer-radius", type=float, required=True, help="Buffer radius (m)")
    parser.add_argument("--decimation-ratio", type=float, required=True, help="Decimation ratio (0-1]")
    parser.add_argument("--point-start", type=int, default=0)
    parser.add_argument("--point-end", type=int, default=None)
    parser.add_argument("--point-step", type=int, default=1)
    parser.add_argument("--featured-dir", type=Path, default=None)
    parser.add_argument("--laz-dir", type=Path, default=None)
    parser.add_argument(
        "--laz-source",
        type=str,
        choices=["geo-mad", "cnig", "both"],
        default="both",
        help="Force which LAZ source to use: geo-mad, cnig, or both (default: both)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--view-mode", type=str, default="3d", choices=["2d","3d","both"])    
    parser.add_argument(
        "--use-all-points",
        action="store_true",
        help="If set, use all LAZ points instead of filtering to ASPRS class 2 (ground)",
    )

    args = parser.parse_args()

    if not 0 < args.buffer_radius:
        parser.error("--buffer-radius must be positive")
    if not 0 < args.decimation_ratio <= 1:
        parser.error("--decimation-ratio must be in (0,1]")

    if args.featured_dir is None:
        args.featured_dir = PROJECT_ROOT / "Doback-Data" / "featured"
    if args.laz_dir is None:
        args.laz_dir = PROJECT_ROOT / "LiDAR-Maps" / "cnig"
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "output" / "visualization"

    geo_mad_dir = PROJECT_ROOT / "LiDAR-Maps" / "geo-mad"
    laz_search_dirs = []
    if args.laz_source in ("geo-mad", "both") and geo_mad_dir.exists():
        laz_search_dirs.append(geo_mad_dir)
    if args.laz_source in ("cnig", "both") and args.laz_dir is not None:
        laz_search_dirs.append(args.laz_dir)

    if not args.featured_dir.exists():
        logger.error(f"Featured data directory not found: {args.featured_dir}")
        sys.exit(1)
    if not any(p.exists() for p in laz_search_dirs):
        logger.error(f"No LAZ directory found. Checked: {laz_search_dirs}")
        sys.exit(1)

    logger.info(f"Visualizing {args.segment_id}")
    logger.info(f"LAZ search dirs: {laz_search_dirs}")

    try:
        segment_data = load_segment_data(
            segment_id=args.segment_id,
            featured_data_dir=args.featured_dir,
            laz_dir=laz_search_dirs,
        )

        segment_data = slice_segment_data(
            segment_data=segment_data,
            point_start=args.point_start,
            point_end=args.point_end,
            point_step=args.point_step,
        )
    except Exception as e:
        logger.error(f"Failed to load segment: {e}")
        sys.exit(1)

    try:
        point_cloud = process_segment_point_cloud(
            segment_data=segment_data,
            buffer_radius=args.buffer_radius,
            decimation_ratio=args.decimation_ratio,
            use_all_points=args.use_all_points,
        )
    except Exception as e:
        logger.error(f"Failed to process point cloud: {e}")
        sys.exit(1)

    try:
        # Create a lightweight SegmentScore when one isn't available
        parts = args.segment_id.split("_")
        device = parts[0] if parts else "UNKNOWN"
        date = parts[1] if len(parts) > 1 else "UNKNOWN"
        segment_score = SegmentScore(
            segment_id=args.segment_id,
            device=device,
            date=date,
            mean_error=0.0,
            std_error=0.0,
            num_samples=0,
            num_points_available=segment_data.num_points,
        )

        output_file = args.output_dir / f"{args.segment_id}_visualization.html"
        html_paths = create_segment_visualization(
            segment_data=segment_data,
            point_cloud_data=point_cloud,
            segment_score=segment_score,
            output_path=output_file,
            view_mode=args.view_mode,
        )
        print("Visualization saved to:")
        for p in html_paths:
            print(p)
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

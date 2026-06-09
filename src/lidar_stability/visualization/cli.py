#!/usr/bin/env python3
"""
Interactive Terrain Impact Visualization CLI

Analyzes model performance by segment and creates interactive maps showing:
- GPS route colored by Stability Index
- LiDAR point cloud decimated for efficient visualization
- Terrain statistics (elevation, roughness, slope)

Usage:
    python -m src.lidar_stability.visualization.cli \
        --model all-devices-no-imu \
        --top-n 5 \
        --buffer-radius 5 \
        --decimation-ratio 0.1
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

from lidar_stability.visualization.segment_ranking import load_and_rank_segments
from lidar_stability.visualization.segment_loader import load_segment_data, slice_segment_data
from lidar_stability.visualization.point_cloud_processor import process_segment_point_cloud
from lidar_stability.visualization.map_builder import create_segment_visualization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _interactive_menu() -> str:
    """
    Display interactive menu for segment selection.
    
    Returns:
        Selected segment ID
    """
    print("\n" + "="*80)
    print("SELECT A SEGMENT TO VISUALIZE")
    print("="*80)
    
    choice = input("\nEnter segment ID (e.g., DOBACK024_20251007_seg28): ").strip()
    return choice


def _show_ranking_table(
    positive: list,
    negative: list,
) -> None:
    """
    Display ranking tables in terminal.
    
    Args:
        positive: List of SegmentScore (positive impact)
        negative: List of SegmentScore (negative impact)
    """
    print("\n" + "="*100)
    print("POSITIVE IMPACT SEGMENTS (Low Error)")
    print("="*100)
    
    if positive:
        for i, score in enumerate(positive, 1):
            print(f"{i}. {score}")
    else:
        print("  No positive impact segments found")
    
    print("\n" + "="*100)
    print("NEGATIVE IMPACT SEGMENTS (High Error)")
    print("="*100)
    
    if negative:
        for i, score in enumerate(negative, 1):
            print(f"{i}. {score}")
    else:
        print("  No negative impact segments found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive terrain impact visualization for LiDAR stability models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
    Examples:
      python -m src.lidar_stability.visualization.cli --buffer-radius 5 --decimation-ratio 0.1
      python -m src.lidar_stability.visualization.cli --model doback-24-no-outliers --top-n 3 --buffer-radius 10 --decimation-ratio 0.05
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="all-devices-no-imu",
        help="Model directory name (default: all-devices-no-imu)",
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top positive/negative segments to show (default: 5)",
    )
    
    parser.add_argument(
        "--buffer-radius",
        type=float,
        required=True,
        help="Buffer radius in meters for extracting LiDAR points (e.g. 5)",
    )
    
    parser.add_argument(
        "--decimation-ratio",
        type=float,
        required=True,
        help="Decimation ratio for point cloud (REQUIRED, e.g., 0.1 for 10 percent)",
    )

    parser.add_argument(
        "--view-mode",
        type=str,
        default="3d",
        choices=["2d", "3d", "both"],
        help="Visualization mode: 3d (default), 2d, or both",
    )

    parser.add_argument(
        "--point-start",
        type=int,
        default=0,
        help="Start index of route points to visualize (inclusive)",
    )

    parser.add_argument(
        "--point-end",
        type=int,
        default=None,
        help="End index of route points to visualize (exclusive). Default: until the end",
    )

    parser.add_argument(
        "--point-step",
        type=int,
        default=1,
        help="Use one point every N points from selected range",
    )
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Path to models directory (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--featured-dir",
        type=Path,
        default=None,
        help="Path to featured data directory (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--laz-dir",
        type=Path,
        default=None,
        help="Path to LAZ directory (auto-detected if not provided)",
    )

    parser.add_argument(
        "--laz-source",
        type=str,
        choices=["geo-mad", "cnig", "both"],
        default="both",
        help="Force which LAZ source to use: geo-mad, cnig, or both (default: both)",
    )

    parser.add_argument(
        "--use-all-points",
        action="store_true",
        help="If set, use all LAZ points instead of filtering to ASPRS class 2 (ground)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for HTML maps (default: output/visualization/)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.buffer_radius:
        parser.error("--buffer-radius must be positive")
    
    if not 0 < args.decimation_ratio <= 1:
        parser.error("--decimation-ratio must be in (0, 1]")

    if args.point_start < 0:
        parser.error("--point-start must be >= 0")

    if args.point_end is not None and args.point_end <= 0:
        parser.error("--point-end must be > 0 when provided")

    if args.point_step <= 0:
        parser.error("--point-step must be > 0")
    
    # Auto-detect paths if not provided
    if args.models_dir is None:
        args.models_dir = PROJECT_ROOT / "output" / "models" / "extra_trees"
    
    if args.featured_dir is None:
        args.featured_dir = PROJECT_ROOT / "Doback-Data" / "featured"
    
    if args.laz_dir is None:
        args.laz_dir = PROJECT_ROOT / "LiDAR-Maps" / "cnig"
    
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "output" / "visualization"
    
    # Build LAZ search order: prefer geo-mad if present, then cnig (args.laz_dir)
    geo_mad_dir = PROJECT_ROOT / "LiDAR-Maps" / "geo-mad"
    laz_search_dirs = []
    # Respect user choice to force a source
    if args.laz_source in ("geo-mad", "both") and geo_mad_dir.exists():
        laz_search_dirs.append(geo_mad_dir)
    if args.laz_source in ("cnig", "both") and args.laz_dir is not None:
        laz_search_dirs.append(args.laz_dir)

    # Verify required paths exist (models and featured) and that at least one LAZ dir exists
    for path, name in [
        (args.models_dir, "models directory"),
        (args.featured_dir, "featured data directory"),
    ]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    if not laz_search_dirs:
        logger.error("No LAZ directories selected (check --laz-source and provided --laz-dir)")
        sys.exit(1)
    if not any(p.exists() for p in laz_search_dirs):
        logger.error(f"No LAZ directory found. Checked: {laz_search_dirs}")
        sys.exit(1)
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Top N: {args.top_n}")
    logger.info(f"  Buffer radius: {args.buffer_radius}m")
    logger.info(f"  Decimation ratio: {args.decimation_ratio}")
    logger.info(f"  View mode: {args.view_mode}")
    logger.info(
        f"  Point selection: start={args.point_start}, end={args.point_end}, step={args.point_step}"
    )
    logger.info(f"  Models directory: {args.models_dir}")
    logger.info(f"  Featured data directory: {args.featured_dir}")
    logger.info(f"  LAZ search dirs: {laz_search_dirs}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    # Step 1: Load and rank segments
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading model predictions and ranking segments")
    logger.info("="*80)
    
    try:
        positive, negative = load_and_rank_segments(
            model_dir=args.models_dir,
            model_name=args.model,
            top_n=args.top_n,
            featured_data_dir=args.featured_dir,
        )
    except Exception as e:
        logger.error(f"Failed to load rankings: {e}")
        sys.exit(1)
    
    # Show ranking tables
    _show_ranking_table(positive, negative)
    
    # Step 2: Interactive segment selection
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Segment Selection")
    logger.info("="*80)
    
    while True:
        segment_id = _interactive_menu()
        
        if not segment_id:
            print("Cancelled.")
            sys.exit(0)
        
        # Find segment in rankings
        all_segments = positive + negative
        segment_score = None
        for score in all_segments:
            if score.segment_id.lower() == segment_id.lower():
                segment_score = score
                break
        
        if segment_score is None:
            # Try loading anyway - it might exist but not be in top N
            print(f"Warning: {segment_id} not in top N rankings, attempting to load...")
            from lidar_stability.visualization.segment_ranking import SegmentScore
            
            # Try to extract device and date from segment_id
            parts = segment_id.split("_")
            if len(parts) >= 2:
                device = parts[0]
                date = parts[1]
                segment_score = SegmentScore(
                    segment_id=segment_id,
                    device=device,
                    date=date,
                    mean_error=0.0,  # Unknown
                    std_error=0.0,
                    num_samples=0,
                    num_points_available=0,
                )
            else:
                print(f"Error: Could not parse segment ID format: {segment_id}")
                continue
        
        print(f"\nSelected: {segment_score}")
        
        # Step 3: Load segment data
        logger.info("\n" + "="*80)
        logger.info(f"STEP 3: Loading segment data ({segment_id})")
        logger.info("="*80)
        
        try:
            segment_data = load_segment_data(
                segment_id=segment_id,
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
            print("Returning to menu...\n")
            continue
        
        # Step 4: Process point cloud
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Processing LiDAR point cloud")
        logger.info("="*80)
        
        try:
            point_cloud = process_segment_point_cloud(
                segment_data=segment_data,
                buffer_radius=args.buffer_radius,
                decimation_ratio=args.decimation_ratio,
                use_all_points=args.use_all_points,
            )
        except Exception as e:
            logger.error(f"Failed to process point cloud: {e}")
            print("Returning to menu...\n")
            continue
        
        # Step 5: Create visualization
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Creating interactive map")
        logger.info("="*80)
        
        try:
            output_file = (
                args.output_dir
                / f"{segment_score.segment_id}_visualization.html"
            )
            
            html_paths = create_segment_visualization(
                segment_data=segment_data,
                point_cloud_data=point_cloud,
                segment_score=segment_score,
                output_path=output_file,
                view_mode=args.view_mode,
            )
            
            print(f"\n{'='*80}")
            print("SUCCESS! Visualization saved to:")
            for path in html_paths:
                print(f"  {path}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            print("Returning to menu...\n")
            continue
        
        # Ask if user wants to visualize another segment
        again = input("Visualize another segment? (y/n): ").strip().lower()
        if again != "y":
            print("\nExiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()

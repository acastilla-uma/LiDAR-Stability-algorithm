#!/usr/bin/env python3
"""
Pipeline completo:
raw -> processed-data -> map-matched -> visualización 2D + 3D.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def find_mapmatched_for_2d(base: str, mapmatched_dir: Path) -> Path:
    seg_files = sorted(mapmatched_dir.glob(f"{base}_seg*.csv"))
    if seg_files:
        return seg_files[0]

    exact = mapmatched_dir / f"{base}.csv"
    if exact.exists():
        return exact

    candidates = sorted(mapmatched_dir.glob(f"{base}*.csv"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No map-matched files found for base '{base}' in {mapmatched_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline from raw data to final 2D and 3D outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", required=True,
                        help="Base name, e.g. DOBACK024_20250929 or DOBACK024_20250929_seg11")
    parser.add_argument("--data-dir", default="Doback-Data")
    parser.add_argument("--processed-dir", default="Doback-Data/processed-data")
    parser.add_argument("--mapmatched-dir", default="Doback-Data/map-matched")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--laz-dir", default="LiDAR-Maps/cnig")
    parser.add_argument("--tif-dir", default="LiDAR-Maps/geo-mad")
    parser.add_argument("--points-sample", type=int, default=700000)
    parser.add_argument("--max-points-2d", type=int, default=300000)
    parser.add_argument("--skip-processing", action="store_true")
    parser.add_argument("--skip-mapmatching", action="store_true")
    parser.add_argument("--skip-terrain-features", action="store_true",
                       help="Skip terrain feature extraction")
    parser.add_argument("--terrain-search-radius", type=float, default=100.0,
                       help="Search radius for terrain features (m)")
    parser.add_argument("--terrain-dem-size", type=int, default=256,
                       help="DEM grid size for feature extraction")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = (repo_root / args.data_dir).resolve()
    processed_dir = (repo_root / args.processed_dir).resolve()
    mapmatched_dir = (repo_root / args.mapmatched_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    laz_dir = (repo_root / args.laz_dir).resolve()
    tif_dir = (repo_root / args.tif_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    mapmatched_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("FULL PIPELINE: RAW -> PROCESSED -> MAP-MATCHED -> 2D + 3D")
    print("=" * 72)
    print(f"Base: {args.base}")

    try:
        if not args.skip_processing:
            run_cmd([
                sys.executable,
                "Scripts/parsers/batch_processor.py",
                "--data-dir", str(data_dir),
                "--output-dir", str(processed_dir),
            ], cwd=repo_root)
        else:
            print("\n[SKIP] raw -> processed")

        if not args.skip_mapmatching:
            run_cmd([
                sys.executable,
                "Scripts/parsers/map_matching.py",
                "--input", str(processed_dir),
                "--output", str(mapmatched_dir),
                "--glob", f"{args.base}*.csv",
            ], cwd=repo_root)
        else:
            print("\n[SKIP] processed -> map-matched")

        # Extract terrain features from map-matched data
        if not args.skip_terrain_features:
            print("\n📊 Extracting terrain features...")
            mapmatch_files = sorted(mapmatched_dir.glob(f"{args.base}*.csv"))
            for mmf in mapmatch_files:
                print(f"  Processing: {mmf.name}")
                try:
                    run_cmd([
                        sys.executable,
                        "Scripts/lidar/compute_route_terrain_features.py",
                        "--mapmatch", str(mmf),
                        "--laz-dir", str(laz_dir),
                        "--output", str(mmf),  # Overwrite with enriched version
                        "--search-radius", str(args.terrain_search_radius),
                        "--dem-size", str(args.terrain_dem_size),
                    ], cwd=repo_root)
                except Exception as e:
                    print(f"  ⚠️ Warning: Terrain feature extraction failed for {mmf.name}: {e}")
                    # Continue with visualization even if feature extraction fails
        else:
            print("\n[SKIP] terrain feature extraction")

        mapmatch_2d = find_mapmatched_for_2d(args.base, mapmatched_dir)
        out_2d = output_dir / f"{args.base}_final_2d.png"
        out_3d = output_dir / f"{args.base}_final_3d.html"

        run_cmd([
            sys.executable,
            "Scripts/visualization/visualize_route_lidar.py",
            "--mapmatch", str(mapmatch_2d),
            "--laz-dir", str(laz_dir),
            "--tif-dir", str(tif_dir),
            "--max-points", str(args.max_points_2d),
            "--output", str(out_2d),
        ], cwd=repo_root)

        run_cmd([
            sys.executable,
            "Scripts/visualization/visualize_3d_interactive.py",
            "--base", args.base,
            "--mapmatch-dir", str(mapmatched_dir),
            "--laz-dir", str(laz_dir),
            "--points-sample", str(args.points_sample),
            "--output", str(out_3d),
        ], cwd=repo_root)

        print("\n✅ Pipeline completado")
        print(f"  2D: {out_2d}")
        print(f"  3D: {out_3d}")
        print(f"  Processed: {processed_dir}")
        print(f"  Map-matched: {mapmatched_dir}")
        return 0

    except Exception as exc:
        print(f"\n❌ ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

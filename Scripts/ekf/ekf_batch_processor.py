"""
Sprint 2: Temporal Matching between GPS and Stability

Pipeline:
- Parse raw GPS and stability logs
- Match GPS and stability by timestamp (temporal alignment)
- Filter anomalies, split by gaps, and export CSVs

Usage:
  python Scripts/ekf/ekf_batch_processor.py
  python Scripts/ekf/ekf_batch_processor.py --tolerance-seconds 1.0 --max-gap-meters 1000
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pyproj import Transformer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers import batch_processor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _find_column(df, candidates):
    """Find column by case-insensitive match."""
    for candidate in candidates:
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    return None


def _get_numeric(row, col, default=0.0):
    """Safe numeric extraction from row."""
    if col is None:
        return default
    val = row.get(col)
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def match_gps_stability(gps_df, stab_df, tolerance_seconds=1.0):
    """
    Simple temporal matching: align GPS and stability records by timestamp.
    
    GPS comes at ~1 Hz (sparse), stability at ~10 Hz (dense).
    For each stability record, find the nearest GPS record within tolerance.
    Output on stability timeline (denser).
    """
    if gps_df is None or stab_df is None or gps_df.empty or stab_df.empty:
        return pd.DataFrame()

    gps_sorted = gps_df.sort_values("timestamp").reset_index(drop=True)
    stab_sorted = stab_df.sort_values("timestamp").reset_index(drop=True)

    # Merge GPS data into stability timeline
    merged = pd.merge_asof(
        stab_sorted,
        gps_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("_stab", "_gps"),
    )

    if merged.empty:
        return pd.DataFrame()

    # Find column names (case-insensitive)
    si_col = _find_column(merged, ["si", "si_mcu", "si_stab", "si_total"])
    lat_col = _find_column(merged, ["lat_gps", "lat"])
    lon_col = _find_column(merged, ["lon_gps", "lon"])
    ax_col = _find_column(merged, ["ax_stab", "ax"])
    ay_col = _find_column(merged, ["ay_stab", "ay"])
    gz_col = _find_column(merged, ["gz_stab", "gz", "gyro_z", "g_z"])

    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)

    fused_rows = []
    for _, row in merged.iterrows():
        t = row["timestamp"]
        if pd.isna(t):
            continue

        # Get GPS position
        lat_val = row.get(lat_col) if lat_col else None
        lon_val = row.get(lon_col) if lon_col else None

        fused_row = {"timestamp": t}

        # Add GPS data if available
        if lat_col and lon_col and not pd.isna(lat_val) and not pd.isna(lon_val):
            fused_row["lat"] = lat_val
            fused_row["lon"] = lon_val
            # Convert to UTM if GPS available
            try:
                x_utm, y_utm = transformer_to_utm.transform(lon_val, lat_val)
                fused_row["x_utm"] = x_utm
                fused_row["y_utm"] = y_utm
            except:
                pass

        # Add stability data if available
        if si_col and not pd.isna(row.get(si_col)):
            fused_row["si"] = row.get(si_col)
        if ax_col:
            fused_row["ax"] = row.get(ax_col)
        if ay_col:
            fused_row["ay"] = row.get(ay_col)
        if gz_col:
            fused_row["gz"] = row.get(gz_col)

        fused_rows.append(fused_row)

    return pd.DataFrame(fused_rows)


def split_segments(df, max_gap_meters=1000, min_points=10):
    """Split data into segments based on distance gaps."""
    if df is None or df.empty or "x_utm" not in df.columns or "y_utm" not in df.columns:
        return []

    x_utm = df["x_utm"].values
    y_utm = df["y_utm"].values
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.diff(x_utm) ** 2 + np.diff(y_utm) ** 2)

    # Find indices where distance exceeds threshold
    gap_indices = np.where(distances > max_gap_meters)[0]
    
    if len(gap_indices) == 0:
        # No gaps - return one segment if large enough
        return [df] if len(df) >= min_points else []

    # Split at gaps
    segments = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        segment = df.iloc[start_idx:end_idx].copy()
        if len(segment) >= min_points:
            segments.append(segment)
        start_idx = end_idx

    # Add final segment
    final_segment = df.iloc[start_idx:].copy()
    if len(final_segment) >= min_points:
        segments.append(final_segment)

    return segments


def process_all(data_dir, output_dir, tolerance_seconds, max_gap_meters=1000):
    """Process all GPS/Stability pairs and export matched CSVs."""
    gps_dir = data_dir / "GPS"
    stab_dir = data_dir / "Stability"

    pairs = batch_processor.build_pairs(gps_dir, stab_dir)
    if not pairs:
        logger.info("No matching GPS/Stability pairs found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "DOBACK TEMPORAL MATCHING REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Max gap for splitting: {max_gap_meters}m",
        f"Matching tolerance: {tolerance_seconds}s",
        "",
    ]

    for key, gps_path, stab_path in pairs:
        try:
            gps_df = batch_processor.parse_gps_file(gps_path)
            stab_df = batch_processor.parse_stability_file(stab_path)

            # Temporal matching (simple, no EKF)
            matched_df = match_gps_stability(gps_df, stab_df, tolerance_seconds)
            if matched_df.empty:
                report_lines.append(f"{key}: no matched rows")
                continue

            # Split into segments
            segments = split_segments(matched_df, max_gap_meters, min_points=10)
            if not segments:
                report_lines.append(f"{key}: no valid segments")
                continue

            # Export segments
            for seg_idx, segment in enumerate(segments, 1):
                suffix = f"_ekf_seg{seg_idx}" if len(segments) > 1 else "_ekf"
                output_file = output_dir / f"{key}{suffix}.csv"
                segment.to_csv(output_file, index=False)
                report_lines.append(f"{key}{suffix}: {len(segment):,} rows")

        except Exception as e:
            report_lines.append(f"{key}: ERROR - {str(e)}")
            logger.error(f"Error processing {key}: {e}")

    report_path = output_dir / "processing_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Done. Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EKF batch processor for DOBACK data")
    parser.add_argument(
        "--data-dir",
        default="Doback-Data",
        help="Base directory with GPS and Stability folders",
    )
    parser.add_argument(
        "--output-dir",
        default="Doback-Data/processed data",
        help="Output directory for processed routes",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=1.0,
        help="Max time difference between GPS and stability (seconds)",
    )
    parser.add_argument(
        "--max-gap-meters",
        type=float,
        default=1000,
        help="Max gap in meters before splitting into segments",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    process_all(
        data_dir,
        output_dir,
        tolerance_seconds=args.tolerance_seconds,
        max_gap_meters=args.max_gap_meters,
    )


if __name__ == "__main__":
    main()

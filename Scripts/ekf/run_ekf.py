"""
End-to-end EKF Execution Script

Usage:
    python scripts/ekf/run_ekf.py <processed_csv> [--output <path>]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ekf.ekf_fusion import ExtendedKalmanFilter
import pyproj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def gps_to_utm(lat, lon, zone=30):
    """
    Convert WGS84 lat/lon to UTM coordinates.
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        zone: UTM zone (30 for Madrid)
        
    Returns:
        (x_utm, y_utm) in meters
    """
    proj_wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    proj_utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
    
    x, y = pyproj.transform(proj_wgs84, proj_utm, lon, lat)
    return x, y


def run_ekf_session(processed_csv: str, output_dir: str = None) -> pd.DataFrame:
    """
    Run EKF on a processed GPS+IMU dataset.
    
    Args:
        processed_csv: Path to processed CSV (matched GPS+IMU)
        output_dir: Optional output directory for results
        
    Returns:
        Trajectory DataFrame with columns [timestamp_utc, x_utm, y_utm, v, yaw, source]
    """
    logger.info("=== EKF Session Processing ===")
    
    # Load processed data
    logger.info(f"Loading processed CSV: {processed_csv}")
    df = pd.read_csv(processed_csv)

    if df.empty:
        logger.error("Processed CSV is empty")
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        logger.error("Processed CSV missing 'timestamp' column")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    if df.empty:
        logger.error("No valid timestamps in processed CSV")
        return pd.DataFrame()
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(state_dim=4, meas_dim_gps=3, wheelbase=3.5)
    
    # Set noise covariances based on GPS quality
    ekf.set_process_noise([0.1, 0.1, 0.5, 0.1])  # Process noise for [x, y, v, psi]
    ekf.set_measurement_noise([2.0, 2.0, 1.0])   # Measurement noise for [x, y, v]
    
    # Convert GPS to UTM
    if "lat" in df.columns and "lon" in df.columns:
        df["x_utm"], df["y_utm"] = zip(*[
            gps_to_utm(lat, lon) for lat, lon in zip(df["lat"], df["lon"])
        ])
    else:
        logger.error("Processed CSV missing lat/lon columns")
        return pd.DataFrame()

    trajectory = []
    prev_time = None

    for _, row in df.iterrows():
        t = row["timestamp"]
        if pd.isna(t):
            continue

        # Time step
        if prev_time is not None:
            dt = (t - prev_time).total_seconds()
            if dt <= 0:
                dt = 0.1
        else:
            dt = 0.1

        # IMU inputs (fallback to 0 if missing)
        ax = (row.get("ax", 0.0) / 1000.0) * 9.81 if not pd.isna(row.get("ax", np.nan)) else 0.0
        ay = (row.get("ay", 0.0) / 1000.0) * 9.81 if not pd.isna(row.get("ay", np.nan)) else 0.0
        psi_dot = np.radians(row.get("gz", 0.0)) if not pd.isna(row.get("gz", np.nan)) else 0.0

        # Predict step
        ekf.predict(ax, ay, psi_dot, dt)

        # Update step with GPS data
        speed_kmh = row.get("speed_kmh", 0.0)
        if pd.isna(speed_kmh):
            speed_kmh = 0.0
        hdop = row.get("hdop", 1.0)
        if pd.isna(hdop):
            hdop = 1.0

        ekf.update(
            row["x_utm"],
            row["y_utm"],
            speed_kmh / 3.6,
            hdop=hdop,
        )

        state = ekf.get_state()
        trajectory.append({
            "timestamp_utc": t,
            "x_utm": state[0],
            "y_utm": state[1],
            "v": state[2],
            "yaw": state[3],
            "source": "merged",
        })

        prev_time = t
    
    trajectory_df = pd.DataFrame(trajectory)
    
    logger.info(f"EKF processed {len(trajectory_df)} samples")
    logger.info(f"  Position range (UTM): X=[{trajectory_df['x_utm'].min():.0f}, {trajectory_df['x_utm'].max():.0f}]")
    logger.info(f"  Position range (UTM): Y=[{trajectory_df['y_utm'].min():.0f}, {trajectory_df['y_utm'].max():.0f}]")
    logger.info(f"  Velocity range: [{trajectory_df['v'].min():.2f}, {trajectory_df['v'].max():.2f}] m/s")
    
    # Save output
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'ekf_trajectory.csv'
        trajectory_df.to_csv(csv_path, index=False)
        logger.info(f"Saved trajectory to {csv_path}")
    
    return trajectory_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EKF on processed GPS+IMU CSV')
    parser.add_argument('processed_csv', help='Processed CSV file with GPS+IMU data')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    trajectory = run_ekf_session(args.processed_csv, args.output)
    
    if not trajectory.empty:
        logger.info("✓ EKF completed successfully")
    else:
        logger.error("✗ EKF failed")
        sys.exit(1)

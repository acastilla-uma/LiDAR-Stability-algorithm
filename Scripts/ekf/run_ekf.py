"""
End-to-end EKF Execution Script

Usage:
    python scripts/ekf/run_ekf.py <gps_file> <imu_file> [--output <path>]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers import parse_gps, parse_imu
from ekf.ekf_fusion import ExtendedKalmanFilter
from ekf.time_sync import calculate_imu_absolute_timestamp
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


def run_ekf_session(gps_file: str, imu_file: str, output_dir: str = None) -> pd.DataFrame:
    """
    Run EKF on a GPS+IMU session.
    
    Args:
        gps_file: Path to GPS data file
        imu_file: Path to IMU data file
        output_dir: Optional output directory for results
        
    Returns:
        Trajectory DataFrame with columns [timestamp_utc, x_utm, y_utm, v, yaw, source]
    """
    logger.info("=== EKF Session Processing ===")
    
    # Parse data
    logger.info(f"Parsing GPS: {gps_file}")
    gps_df = parse_gps(gps_file)
    
    logger.info(f"Parsing IMU: {imu_file}")
    imu_df = parse_imu(imu_file)
    
    if gps_df.empty or imu_df.empty:
        logger.error("One or both dataframes are empty")
        return pd.DataFrame()
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(state_dim=4, meas_dim_gps=3, wheelbase=3.5)
    
    # Set noise covariances based on GPS quality
    ekf.set_process_noise([0.1, 0.1, 0.5, 0.1])  # Process noise for [x, y, v, psi]
    ekf.set_measurement_noise([2.0, 2.0, 1.0])   # Measurement noise for [x, y, v]
    
    # Get session timing reference from GPS
    session_start_utc = gps_df['timestamp_utc'].min()
    
    # Convert GPS to UTM
    gps_df['x_utm'], gps_df['y_utm'] = zip(*[
        gps_to_utm(lat, lon) for lat, lon in zip(gps_df['lat'], gps_df['lon'])
    ])
    
    # Convert IMU timestamps to absolute
    imu_df['timestamp_utc'] = calculate_imu_absolute_timestamp(imu_df, session_start_utc)
    
    # Merge timelines
    logger.info("Merging GPS and IMU timelines...")
    all_timestamps = pd.concat([
        gps_df['timestamp_utc'],
        imu_df['timestamp_utc']
    ]).drop_duplicates().sort_values()
    
    trajectory = []
    gps_idx = 0
    imu_idx = 0
    
    for t in all_timestamps:
        # Determine if this is a GPS update or IMU-only step
        is_gps = t in gps_df['timestamp_utc'].values
        
        if is_gps and gps_idx < len(gps_df):
            # GPS update available
            gps_row = gps_df[gps_df['timestamp_utc'] == t].iloc[0]
            
            # Compute acceleration from GPS velocity change (rough estimate)
            if len(trajectory) > 0:
                dt = (t - trajectory[-1]['timestamp_utc']).total_seconds()
                if dt > 0:
                    dv = gps_row['speed_kmh'] / 3.6 - trajectory[-1]['v']
                    ax = dv / dt
                else:
                    ax = 0
            else:
                ax = 0
            
            # Predict step with IMU data (dummy for now)
            ekf.predict(ax, 0, 0, 1.0)
            
            # Update step with GPS
            ekf.update(
                gps_row['x_utm'],
                gps_row['y_utm'],
                gps_row['speed_kmh'] / 3.6,
                hdop=gps_row['hdop']
            )
            
            state = ekf.get_state()
            trajectory.append({
                'timestamp_utc': t,
                'x_utm': state[0],
                'y_utm': state[1],
                'v': state[2],
                'yaw': state[3],
                'source': 'gps'
            })
            gps_idx += 1
        else:
            # IMU-only step (high frequency)
            imu_row = imu_df[imu_df['timestamp_utc'] == t]
            if len(imu_row) > 0:
                imu_row = imu_row.iloc[0]
                
                # Convert IMU accelerations (assuming raw LSB, ~1g = 1000)
                ax = imu_row['ax'] / 1000.0 * 9.81
                ay = imu_row['ay'] / 1000.0 * 9.81
                psi_dot = np.radians(imu_row['gz'])  # Gyro Z to rad/s (assume degree/s input)
                
                # Time step
                if len(trajectory) > 0:
                    dt = (t - trajectory[-1]['timestamp_utc']).total_seconds()
                else:
                    dt = 0.1
                
                # Predict only
                ekf.predict(ax, ay, psi_dot, dt)
                
                state = ekf.get_state()
                trajectory.append({
                    'timestamp_utc': t,
                    'x_utm': state[0],
                    'y_utm': state[1],
                    'v': state[2],
                    'yaw': state[3],
                    'source': 'imu'
                })
                imu_idx += 1
    
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
    parser = argparse.ArgumentParser(description='Run EKF on GPS+IMU data')
    parser.add_argument('gps_file', help='GPS data file')
    parser.add_argument('imu_file', help='IMU data file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    trajectory = run_ekf_session(args.gps_file, args.imu_file, args.output)
    
    if not trajectory.empty:
        logger.info("✓ EKF completed successfully")
    else:
        logger.error("✗ EKF failed")
        sys.exit(1)

"""
IMU / Estabilidad Data Parser

Reads ESTABILIDAD_DOBACK*.txt files and returns clean pandas DataFrame.
Handles timestamp markers and restart lines.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def parse_imu(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """
    Parse IMU/Stability data from DOBACK Stability file.
    
    Args:
        filepath: Path to ESTABILIDAD_DOBACK*.txt file
        verbose: Print debug info
        
    Returns:
        pd.DataFrame with columns:
        [ax, ay, az, gx, gy, gz, roll_deg, pitch_deg, yaw_deg, t_us, si_mcu,
         accmag, microsds, k3, device_id, session]
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    metadata = {}
    rows = []
    current_timestamp_us = 0  # Keep track of microsecond counter
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        logger.warning(f"File {filepath} has fewer than 2 lines. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Parse metadata (line 1)
    meta_line = lines[0].strip()
    parts = [x.strip() for x in meta_line.split(';')]
    if len(parts) >= 4:
        metadata['sensor_type'] = parts[0].strip()
        # parts[1] is datetime
        metadata['device_id'] = parts[2].strip()
        metadata['session'] = int(parts[3].strip()) if parts[3].strip() else 0
    
    # Expected header at line 2
    # ax; ay; az; gx; gy; gz; roll; pitch; yaw; timeantwifi; usciclo1; usciclo2; usciclo3; usciclo4; usciclo5; si; accmag; microsds; k3
    
    for i, line in enumerate(lines[2:], start=2):
        line = line.strip()
        if not line:
            continue
        
        # Skip header line
        if line.lower().startswith('ax;'):
            continue
        
        # Skip restart/metadata lines
        if line.upper().startswith('ESTABILIDAD'):
            continue
        
        # Check if this is a timestamp marker line (HH:MM:SSAM or HH:MM:SSPM)
        if line.upper().endswith('AM') or line.upper().endswith('PM'):
            logger.debug(f"Skipping timestamp marker at line {i}: {line}")
            continue
        
        # Parse data row
        try:
            cols = [x.strip() for x in line.split(';')]
        except:
            logger.debug(f"Failed to parse line {i}: {line}")
            continue
        
        if len(cols) < 19:
            logger.debug(f"Line {i} has fewer than 19 columns ({len(cols)}): {line[:60]}")
            continue
        
        try:
            ax = float(cols[0])
            ay = float(cols[1])
            az = float(cols[2])
            gx = float(cols[3])
            gy = float(cols[4])
            gz = float(cols[5])
            roll = float(cols[6])
            pitch = float(cols[7])
            yaw = float(cols[8])
            t_us = float(cols[9])  # timeantwifi in microseconds
            usciclo1 = float(cols[10])
            usciclo2 = float(cols[11])
            usciclo3 = float(cols[12])
            usciclo4 = float(cols[13])
            usciclo5 = float(cols[14])
            si = float(cols[15])  # Stability index from MCU
            accmag = float(cols[16])  # Acceleration magnitude
            microsds = float(cols[17])  # SD card write time
            k3 = float(cols[18])  # Unknown parameter
            
            # Sanity checks
            if not (-100 <= roll <= 100):  # Roll in reasonable range
                continue
            if not (-100 <= pitch <= 100):  # Pitch in reasonable range
                continue
            if not (0 <= si <= 2.0):  # SI should be in [0, 2]
                continue
            
            rows.append({
                'ax': ax,
                'ay': ay,
                'az': az,
                'gx': gx,
                'gy': gy,
                'gz': gz,
                'roll_deg': roll,
                'pitch_deg': pitch,
                'yaw_deg': yaw,
                't_us': t_us,
                'usciclo1': usciclo1,
                'usciclo2': usciclo2,
                'usciclo3': usciclo3,
                'usciclo4': usciclo4,
                'usciclo5': usciclo5,
                'si_mcu': si,
                'accmag': accmag,
                'microsds': microsds,
                'k3': k3,
                'device_id': metadata.get('device_id', 'UNKNOWN'),
                'session': metadata.get('session', 0)
            })
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse line {i}: {line[:60]}. Error: {e}")
            continue
    
    if not rows:
        logger.warning(f"No valid IMU rows parsed from {filepath}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Sort by timestamp
    df = df.sort_values('t_us').reset_index(drop=True)
    
    # Calculate dt between consecutive rows to verify ~10 Hz
    if len(df) > 1:
        dt_us = df['t_us'].diff().dropna()
        dt_ms = dt_us / 1000.0
        median_dt_ms = dt_ms.median()
        mean_dt_ms = dt_ms.mean()
        
        logger.info(f"Parsed {len(df)} valid IMU records from {filepath}")
        logger.info(f"  Device: {metadata.get('device_id', 'UNKNOWN')}, Session: {metadata.get('session', 0)}")
        logger.info(f"  Time span: {dt_us.sum() / 1e6:.1f} seconds (~{len(df) / (dt_us.sum() / 1e6):.1f} Hz)")
        logger.info(f"  Median dt: {median_dt_ms:.2f} ms (expected ~100 ms for 10 Hz)")
        logger.info(f"  Roll range: [{df['roll_deg'].min():.2f}°, {df['roll_deg'].max():.2f}°]")
        logger.info(f"  Pitch range: [{df['pitch_deg'].min():.2f}°, {df['pitch_deg'].max():.2f}°]")
        logger.info(f"  SI range: [{df['si_mcu'].min():.3f}, {df['si_mcu'].max():.3f}]")
    
    return df

"""
GPS Data Parser

Reads GPS_DOBACK*.txt files and returns clean pandas DataFrame.
Handles data corruption and missing values.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def parse_gps(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """
    Parse GPS data from DOBACK GPS file.
    
    Args:
        filepath: Path to GPS_DOBACK*.txt file
        verbose: Print debug info
        
    Returns:
        pd.DataFrame with columns: 
        [timestamp_utc, lat, lon, alt, hdop, fix, num_sats, speed_kmh, 
         device_id, session]
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    metadata = {}
    rows = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        logger.warning(f"File {filepath} has fewer than 2 lines. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Parse metadata (line 1)
    meta_line = lines[0].strip()
    parts = meta_line.split(';')
    if len(parts) >= 5:
        metadata['sensor_type'] = parts[0].strip()
        # parts[1] is datetime
        metadata['device_id'] = parts[2].strip()
        metadata['session'] = int(parts[4].strip()) if parts[4].strip() else 0
    
    # Parse data rows (skip header at line 2 onwards)
    for i, line in enumerate(lines[2:], start=2):
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a "no-fix" row
        if 'sin datos GPS' in line.lower():
            continue
        
        # Split by comma
        try:
            cols = [x.strip() for x in line.split(',')]
        except:
            logger.debug(f"Failed to parse line {i}: {line}")
            continue
        
        if len(cols) < 10:
            logger.debug(f"Line {i} has fewer than 10 columns: {line}")
            continue
        
        try:
            # Expected format:
            # HoraRaspberry, Fecha, Hora(GPS), Lat, Lon, Alt, HDOP, Fix, NumSats, Velocidad
            hora_rpi = cols[0]
            fecha = cols[1]
            hora_gps = cols[2]
            lat_str = cols[3]
            lon_str = cols[4]
            alt_str = cols[5]
            hdop_str = cols[6]
            fix_str = cols[7]
            num_sats_str = cols[8]
            speed_str = cols[9]
            
            # Parse latitude (handle truncation: "0.5351815" should be "40.5351815")
            lat = float(lat_str)
            if 36 <= lat <= 44:  # Valid range for Madrid region
                pass
            elif 0 <= lat <= 1:  # Likely truncated
                lat += 40
            else:
                continue  # Invalid latitude
            
            # Parse longitude
            lon = float(lon_str)
            if not (-10 <= lon <= 5):  # Valid range for Peninsula
                continue
            
            # Parse altitude
            alt = float(alt_str)
            
            # Parse HDOP
            hdop = float(hdop_str)
            if hdop > 10:  # Poor GPS accuracy
                continue
            
            # Parse fix
            fix = int(fix_str)
            if fix not in [1, 2]:  # 1=GPS fix, 2=DGPS
                continue
            
            # Parse satellites
            num_sats = int(num_sats_str)
            
            # Parse speed
            speed = float(speed_str)
            if speed > 200:  # Absurd speed (sanity check)
                continue
            
            # Parse timestamp
            try:
                dt_str = f"{fecha} {hora_gps}"
                timestamp = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            except:
                logger.debug(f"Failed to parse timestamp: {dt_str}")
                continue
            
            rows.append({
                'timestamp_utc': timestamp,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'hdop': hdop,
                'fix': fix,
                'num_sats': num_sats,
                'speed_kmh': speed,
                'device_id': metadata.get('device_id', 'UNKNOWN'),
                'session': metadata.get('session', 0)
            })
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse line {i}: {line}. Error: {e}")
            continue
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        logger.warning(f"No valid GPS rows parsed from {filepath}")
        return df
    
    # Sort by timestamp
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    logger.info(f"Parsed {len(df)} valid GPS records from {filepath}")
    logger.info(f"  Device: {metadata.get('device_id', 'UNKNOWN')}, Session: {metadata.get('session', 0)}")
    logger.info(f"  Lat range: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
    logger.info(f"  Lon range: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
    logger.info(f"  Speed range: [{df['speed_kmh'].min():.2f}, {df['speed_kmh'].max():.2f}] km/h")
    
    return df

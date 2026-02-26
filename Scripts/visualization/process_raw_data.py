"""
Script para procesar datos RAW GPS e IMU y generar archivos formateados
para visualización.

Este script convierte los archivos RAW de Doback-Data/ al formato
esperado por las herramientas de visualización.

Autor: Alex Castilla
Fecha: 2025-02-24
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path

try:
    from pyproj import Transformer
except ImportError:
    print("Advertencia: pyproj no disponible, usando conversión aproximada")
    Transformer = None

# Parámetros del vehículo (DOBACK024)
VEHICLE_MASS = 18000  # kg
TRACK_WIDTH = 2.48    # m
CG_HEIGHT = 1.85      # m
PHI_C = 33.79         # ángulo crítico de vuelco (grados)


def parse_raw_gps_file(filepath):
    """
    Parsea archivo GPS RAW del formato Doback.
    
    Formato esperado:
    GPS;14/08/2025 10:33:28;DOBACK027;34;0
    HoraRaspberry,Fecha,Hora(GPS),Latitud,Longitud,Altitud,HDOP,Fix,NumSats,Velocidad(km/h)
    Hora Raspberry-10:33:29,14/08/2025,Hora GPS-08:33:27,sin datos GPS
    """
    print(f"Parseando GPS RAW: {filepath}")
    
    data = []
    device_id = None
    session_date = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Primera línea: metadata
        if lines:
            header_parts = lines[0].strip().split(';')
            if len(header_parts) >= 3:
                session_date = header_parts[1]
                device_id = header_parts[2]
        
        # Saltar header de columnas (línea 2)
        for line_num, line in enumerate(lines[2:], start=3):
            parts = line.strip().split(',')
            
            # Verificar que no sea "sin datos GPS"
            if len(parts) < 4 or 'sin datos GPS' in line:
                continue
            
            try:
                # Hora Raspberry-10:35:12
                hora_rasp = parts[0].split('-')[1] if '-' in parts[0] else parts[0]
                fecha = parts[1]  # 14/08/2025
                hora_gps = parts[2].split('-')[1] if '-' in parts[2] else parts[2]
                
                # Coordenadas
                lat = float(parts[3])
                lon = float(parts[4])
                alt = float(parts[5])
                hdop = float(parts[6])
                fix = int(parts[7])
                num_sats = int(parts[8])
                speed_kmh = float(parts[9])
                
                # Convertir a timestamp UTC
                dt_str = f"{fecha} {hora_rasp}"
                try:
                    dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
                    timestamp_us = int(dt.timestamp() * 1e6)
                except:
                    timestamp_us = line_num * 1000000
                
                # Convertir lat/lon a UTM
                x_utm, y_utm = lat, lon  # Fallback: usar directamente
                
                if Transformer is not None:
                    try:
                        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
                        x_utm, y_utm = transformer.transform(lon, lat)
                    except:
                        pass
                
                record = {
                    'timestamp': timestamp_us,
                    'utc_datetime': dt_str,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'x_utm': x_utm,
                    'y_utm': y_utm,
                    'speed': speed_kmh / 3.6,  # m/s
                    'heading': 0.0,
                    'hdop': hdop,
                    'fix': fix,
                    'num_sats': num_sats
                }
                data.append(record)
                
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    print(f"  ✓ {len(df)} registros GPS parseados")
    return df


def calculate_si_static(roll_deg):
    """Calcula SI estático a partir del ángulo de roll."""
    # SI_static = sin(phi_c - roll) / sin(phi_c)
    # donde phi_c es el ángulo crítico de vuelco
    phi_c_rad = np.radians(PHI_C)
    roll_rad = np.radians(roll_deg)
    
    numerator = np.sin(phi_c_rad - roll_rad)
    denominator = np.sin(phi_c_rad)
    
    si_static = numerator / denominator if denominator != 0 else 1.0
    return np.clip(si_static, 0.1, 1.5)  # Limitar a [0.1, 1.5] (1.0 = estable)


def parse_raw_stability_file(filepath):
    """
    Parsea archivo de estabilidad RAW del formato Doback.
    
    Formato:
    ESTABILIDAD; 25/08/2025 12:00:19; DOBACK024;188;
    ax; ay; az; gx; gy; gz; roll; pitch; yaw; timeantwifi; usciclo1; ...
    45.14;  61.24; 1005.65;   6.74; -33.69; -29.58;  -2.57;   3.50;   0.00; ...
    """
    print(f"Parseando Estabilidad RAW: {filepath}")
    
    data = []
    device_id = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Primera línea: metadata
        if lines:
            header_parts = lines[0].strip().split(';')
            if len(header_parts) >= 3:
                device_id = header_parts[2].strip()
        
        # Segunda línea: headers (saltar)
        # Tercera línea en adelante: datos
        for line_num, line in enumerate(lines[2:], start=3):
            parts = [p.strip() for p in line.strip().split(';')]
            
            if len(parts) < 10:
                continue
            
            try:
                ax = float(parts[0])
                ay = float(parts[1])
                az = float(parts[2])
                gx = float(parts[3])
                gy = float(parts[4])
                gz = float(parts[5])
                roll = float(parts[6])
                pitch = float(parts[7])
                yaw = float(parts[8])
                time_ant = float(parts[9])
                
                # Calcular SI estático desde roll
                si_static = calculate_si_static(roll)
                
                # Aceleración lateral en m/s²
                accel_lat = ay / 100.0
                
                # ΔSI ≈ (a_lat * Hg) / (g * S/2)
                g = 9.81
                delta_si = (accel_lat * CG_HEIGHT) / (g * TRACK_WIDTH / 2.0)
                
                # Limitar valores extremos
                if abs(delta_si) > 1.0:
                    delta_si = np.sign(delta_si) * 1.0
                
                si_dynamic = delta_si
                si_total = si_static + si_dynamic
                si_total = np.clip(si_total, 0.1, 1.5)  # 1.0 = ESTABLE
                
                # Clasificar riesgo: SI=1.0 es UMBRAL SEGURO
                if si_total < 0.7:
                    risk = 'CRITICAL'  # Muy peligroso
                elif si_total < 0.9:
                    risk = 'HIGH'      # Peligroso
                elif si_total < 1.0:
                    risk = 'MEDIUM'    # Alerta
                else:
                    risk = 'LOW'       # Seguro
                
                # Timestamp
                timestamp_us = int(time_ant) if time_ant > 1e9 else line_num * 100000
                
                # Speed (simplificado)
                speed = min(abs(ax) / 100.0, 40.0)
                
                record = {
                    'timestamp': timestamp_us,
                    'utc_datetime': f"2025-08-25 12:00:{line_num % 60:02d}",
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'accel_lat': accel_lat,
                    'accel_lon': ax / 100.0,
                    'accel_vert': az / 100.0,
                    'gyro_x': gx,
                    'gyro_y': gy,
                    'gyro_z': gz,
                    'speed': speed,
                    'si_static': si_static,
                    'si_dynamic': si_dynamic,
                    'si_total': si_total,
                    'rollover_risk': risk
                }
                data.append(record)
                
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    print(f"  ✓ {len(df)} registros de estabilidad parseados")
    return df


def save_processed_gps(df, output_path):
    """Guarda GPS procesado en formato esperado por visualización."""
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            line = f"{int(row['timestamp'])};{row['utc_datetime']};{row['latitude']:.6f};"
            line += f"{row['longitude']:.6f};{row['altitude']:.2f};{row['x_utm']:.2f};"
            line += f"{row['y_utm']:.2f};{row['speed']:.2f};{row['heading']:.2f}\n"
            f.write(line)
    
    print(f"  ✓ GPS procesado guardado en: {output_path}")


def save_processed_stability(df, output_path):
    """Guarda estabilidad procesada en formato esperado por visualización."""
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            line = f"{int(row['timestamp'])};{row['utc_datetime']};"
            line += f"{row['roll']:.2f};{row['pitch']:.2f};{row['accel_lat']:.2f};"
            line += f"{row['accel_lon']:.2f};{row['speed']:.2f};"
            line += f"{row['si_static']:.3f};{row['si_dynamic']:.3f};"
            line += f"{row['si_total']:.3f};{row['rollover_risk']}\n"
            f.write(line)
    
    print(f"  ✓ Estabilidad procesada guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Procesar datos RAW GPS e IMU para visualización')
    parser.add_argument('--gps', help='Ruta al archivo GPS RAW')
    parser.add_argument('--stability', help='Ruta al archivo de estabilidad RAW')
    parser.add_argument('--output-dir', default='output/processed',
                       help='Directorio de salida para archivos procesados')
    parser.add_argument('--all', action='store_true',
                       help='Procesar todos los archivos en Doback-Data/')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # Procesar todos los archivos
        project_root = Path(__file__).parent.parent.parent
        gps_dir = project_root / "Doback-Data" / "GPS"
        stab_dir = project_root / "Doback-Data" / "Stability"
        
        # Procesar GPS
        for gps_file in gps_dir.glob("*.txt"):
            try:
                gps_df = parse_raw_gps_file(gps_file)
                if not gps_df.empty:
                    output_file = output_dir / f"GPS_{gps_file.stem}_processed.txt"
                    save_processed_gps(gps_df, output_file)
            except Exception as e:
                print(f"  ✗ Error procesando {gps_file.name}: {e}")
        
        # Procesar Estabilidad
        for stab_file in stab_dir.glob("*.txt"):
            try:
                stab_df = parse_raw_stability_file(stab_file)
                if not stab_df.empty:
                    output_file = output_dir / f"STABILITY_{stab_file.stem}_processed.txt"
                    save_processed_stability(stab_df, output_file)
            except Exception as e:
                print(f"  ✗ Error procesando {stab_file.name}: {e}")
    
    else:
        # Procesar archivos individuales
        if args.gps:
            gps_df = parse_raw_gps_file(args.gps)
            if not gps_df.empty:
                output_file = output_dir / f"GPS_processed.txt"
                save_processed_gps(gps_df, output_file)
        
        if args.stability:
            stab_df = parse_raw_stability_file(args.stability)
            if not stab_df.empty:
                output_file = output_dir / f"STABILITY_processed.txt"
                save_processed_stability(stab_df, output_file)
    
    print("\n✓ Procesamiento completado")
    print(f"  Archivos guardados en: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

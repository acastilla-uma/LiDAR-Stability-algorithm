"""
Script para generar datos de prueba sintéticos GPS + Estabilidad para visualización.

Genera trayectorias realistas con datos de estabilidad integrados.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_synthetic_data(n_points=500, vehicle_speed=15):
    """
    Genera datos sintéticos GPS + Estabilidad.
    
    Args:
        n_points: Número de puntos en la trayectoria
        vehicle_speed: Velocidad del vehículo en m/s
    
    Returns:
        Tuple (gps_df, stability_df) con datos sincronizados
    """
    print(f"Generando {n_points} puntos de prueba...")
    
    # Parámetros
    PHI_C = 33.79  # ángulo crítico de vuelco (grados)
    g = 9.81
    
    # Trayectoria (espiral/curve en coordenadas UTM30N)
    # Centro: Madrid aproximadamente
    x_center, y_center = 444000, 4486000
    
    # Generar trayectoria con curvas
    t = np.linspace(0, 4*np.pi, n_points)
    radius = np.linspace(500, 1500, n_points)  # Espiral
    
    x_utm = x_center + radius * np.cos(t)
    y_utm = y_center + radius * np.sin(t)
    
    # Convertir a lat/lon (aproximado)
    # ETRS89 UTM30N: 1 grado ≈ 111 km
    lat_center, lon_center = 40.4168, -3.7038  # Madrid
    lat = lat_center + (y_utm - y_center) / 111000
    lon = lon_center + (x_utm - x_center) / (111000 * np.cos(np.radians(lat_center)))
    
    # Elevación realista
    altitude = 700 + 50 * np.sin(2*t) + np.random.randn(n_points) * 5
    
    # Timestamps (crear secuencia realista)
    start_time = datetime(2025, 8, 14, 10, 33, 28)
    dt_sec = 1 / (vehicle_speed / 100)  # Determinar dt según velocidad y distancia
    timestamps = [int((start_time + timedelta(seconds=i*dt_sec)).timestamp() * 1e6) 
                  for i in range(n_points)]
    
    # Velocidad (variable con ruido)
    speed = vehicle_speed + 2 * np.sin(t) + np.random.randn(n_points) * 0.5
    speed = np.clip(speed, 0, 40)  # [0-40] m/s
    
    # Heading (dirección de movimiento)
    heading = np.degrees(np.arctan2(np.gradient(y_utm), np.gradient(x_utm)))
    heading[0] = 0
    
    # Datos GPS
    gps_df = pd.DataFrame({
        'timestamp': timestamps,
        'utc_datetime': [start_time + timedelta(seconds=i*dt_sec) for i in range(n_points)],
        'latitude': lat,
        'longitude': lon,
        'altitude': altitude,
        'x_utm': x_utm,
        'y_utm': y_utm,
        'speed': speed,
        'heading': heading
    })
    
    # ============================================================
    # Datos de Estabilidad con correlación a trayectoria
    # ============================================================
    
    # Roll (sigue a la curvatura de la trayectoria)
    curvature = np.abs(np.gradient(heading))
    curvature[0] = 0
    curvature = np.convolve(curvature, np.ones(5)/5, mode='same')  # Suavizar
    
    roll_deg = 5 * curvature + np.random.randn(n_points) * 0.5
    roll_deg = np.clip(roll_deg, -33, 33)
    
    # Pitch (seguir elevación)
    pitch_deg = np.arctan(np.gradient(altitude) / 100) * (180/np.pi) + np.random.randn(n_points) * 0.3
    pitch_deg = np.clip(pitch_deg, -20, 20)
    
    # Aceleración lateral (derivada de roll y velocidad)
    accel_lat = (roll_deg / 30) * g + np.random.randn(n_points) * 0.2
    accel_lat = np.clip(accel_lat, -5, 5)
    
    # Aceleración longitudinal
    accel_lon = np.gradient(speed) / 0.1 + np.random.randn(n_points) * 0.1
    accel_lon = np.clip(accel_lon, -8, 8)
    
    # Aceleración vertical
    accel_vert = 9.81 + np.random.randn(n_points) * 0.3
    
    # Calcular SI estático
    PHI_C_RAD = np.radians(PHI_C)
    roll_rad = np.radians(roll_deg)
    si_static = np.sin(PHI_C_RAD - roll_rad) / np.sin(PHI_C_RAD)
    si_static = np.clip(si_static, 0.1, 2.0)
    
    # SI dinámico (comportamiento realista)
    TRACK_WIDTH = 2.48
    CG_HEIGHT = 1.85
    si_dynamic = (accel_lat * CG_HEIGHT) / (g * TRACK_WIDTH / 2)
    si_dynamic = np.clip(si_dynamic, -1, 1)
    
    # SI total
    si_total = si_static + 0.3 * si_dynamic
    si_total = np.clip(si_total, 0.1, 2.0)
    
    # Clasificar riesgo
    def get_risk(si):
        if si < 0.7:
            return 'LOW'
        elif si < 0.9:
            return 'MEDIUM'
        elif si < 1.1:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    rollover_risk = [get_risk(si) for si in si_total]
    
    # Datos de Estabilidad
    stability_df = pd.DataFrame({
        'timestamp': timestamps,
        'utc_datetime': gps_df['utc_datetime'],
        'roll': roll_deg,
        'pitch': pitch_deg,
        'yaw': heading,
        'accel_lat': accel_lat,
        'accel_lon': accel_lon,
        'accel_vert': accel_vert,
        'speed': speed,
        'si_static': si_static,
        'si_dynamic': si_dynamic,
        'si_total': si_total,
        'rollover_risk': rollover_risk
    })
    
    print(f"  GPS: {len(gps_df)} puntos, rango: {gps_df['x_utm'].min():.1f}-{gps_df['x_utm'].max():.1f}")
    print(f"  ESTAB: {len(stability_df)} puntos")
    print(f"  SI rango: {si_total.min():.3f}-{si_total.max():.3f}")
    print(f"  Riesgos: {pd.Series(rollover_risk).value_counts().to_dict()}")
    
    return gps_df, stability_df


def save_data(gps_df, stability_df, gps_path, stability_path):
    """Guardar datos en formato compatible con visualización."""
    
    # GPS
    with open(gps_path, 'w') as f:
        for _, row in gps_df.iterrows():
            line = f"{int(row['timestamp'])};{row['utc_datetime']};"
            line += f"{row['latitude']:.6f};{row['longitude']:.6f};"
            line += f"{row['altitude']:.2f};{row['x_utm']:.2f};"
            line += f"{row['y_utm']:.2f};{row['speed']:.2f};{row['heading']:.2f}\n"
            f.write(line)
    print(f"GPS guardado: {gps_path}")
    
    # Estabilidad
    with open(stability_path, 'w') as f:
        for _, row in stability_df.iterrows():
            line = f"{int(row['timestamp'])};{row['utc_datetime']};"
            line += f"{row['roll']:.2f};{row['pitch']:.2f};"
            line += f"{row['accel_lat']:.2f};{row['accel_lon']:.2f};"
            line += f"{row['speed']:.2f};{row['si_static']:.3f};"
            line += f"{row['si_dynamic']:.3f};{row['si_total']:.3f};"
            line += f"{row['rollover_risk']}\n"
            f.write(line)
    print(f"Estabilidad guardada: {stability_path}")


if __name__ == '__main__':
    # Generar datos
    gps_df, stability_df = generate_synthetic_data(n_points=1000, vehicle_speed=15)
    
    # Guardar
    output_dir = Path('output/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data(gps_df, stability_df,
              output_dir / 'GPS_synthetic.txt',
              output_dir / 'STABILITY_synthetic.txt')
    
    print(f"\nDatos de prueba generados en {output_dir}/")
    print("Puedes usarlos así:")
    print(f'  python Scripts/visualization/visualize_all.py "LiDAR-Maps\\cnig\\" "output\\synthetic\\GPS_synthetic.txt" "output\\synthetic\\STABILITY_synthetic.txt" --sample 0.01')

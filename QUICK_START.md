# 🚀 Quick Start Guide

**LiDAR-Stability Algorithm - PIML Traversability Mapping**

Implementación Sprints 1-2 COMPLETADOS. Puedes empezar a usar los módulos inmediatamente.

---

## ⚡ 5-Minuto Quick Start

### 1. Verificar Instalación

```bash
cd LiDAR-Stability-algorithm
pip install -r requirements.txt  # Ya ejecutado, pero verifica

# Comprobar
python -c "import numpy, pandas, scipy; print('✓ OK')"
```

### 2. Parsear Datos Brutos

**GPS:**
```python
from scripts.parsers import parse_gps

gps_df = parse_gps('Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt')
print(f"Parsed {len(gps_df)} GPS records")
print(gps_df[['timestamp_utc', 'lat', 'lon', 'speed_kmh']].head())
```

**IMU/Estabilidad:**
```python
from scripts.parsers import parse_imu

imu_df = parse_imu('Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt')
print(f"Parsed {len(imu_df)} IMU records @ ~10 Hz")
print(imu_df[['roll_deg', 'pitch_deg', 'si_mcu']].head())
```

### 3. Calcular Estabilidad Física

```python
from scripts.physics import StabilityEngine

# Cargar configuración del vehículo
engine = StabilityEngine('scripts/config/vehicle.yaml')

# Ángulo crítico de vuelco
phi_c = engine.critical_angle(degrees=True)
print(f"Critical rollover angle: {phi_c:.2f}°")

# Calcular SI para un ángulo de roll
roll_deg = 15.0
si = engine.si_static_from_deg(roll_deg)
print(f"SI at {roll_deg}° roll: {si:.3f}")
# SI < 1.0 = estable, SI >= 1.0 = peligro
```

### 4. Generar Ground Truth

```python
from scripts.parsers import parse_imu
from scripts.physics import StabilityEngine
from scripts.pipeline import build_ground_truth, export_ground_truth

# Parsear datos reales
imu_df = parse_imu('Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt')
engine = StabilityEngine('scripts/config/vehicle.yaml')

# Construir ground truth con ΔSI como target
gt_df = build_ground_truth(imu_df, engine)
print(f"Ground truth: {len(gt_df)} samples")
print(gt_df[['si_real', 'si_static', 'delta_si']].describe())

# Exportar para análisis posterior
export_ground_truth(gt_df, 'output/ground_truth.csv')
```

### 5. Ejecutar EKF Sensor Fusion

```bash
# Fusionar GPS (1 Hz) + IMU (10 Hz) en trayectoria continua
python scripts/ekf/run_ekf.py \
  Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt \
  Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt \
  --output output/

# Output: output/ekf_trajectory.csv
```

---

## 🧪 Ejecutar Tests

```bash
# Sprint 1: Parsers + Physics
pytest scripts/tests/test_sprint1.py -v

# Sprint 2: EKF
pytest scripts/tests/test_sprint2.py -v

# Todos (Sprint 1-2)
pytest scripts/tests/test_sprint1.py scripts/tests/test_sprint2.py -v

# Resumen rápido
pytest --collect-only -q scripts/tests/
```

**Esperado:** 25/25 tests PASSED ✅

---

## 📊 Explorar Resultados

```python
import pandas as pd
import matplotlib.pyplot as plt

# Leer ground truth
gt = pd.read_csv('output/ground_truth.csv')

# Verificar distribución de ΔSI
print(f"ΔSI mean: {gt['delta_si'].mean():.3f}")
print(f"ΔSI std: {gt['delta_si'].std():.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# SI distribution
axes[0].hist(gt['si_real'], bins=50, alpha=0.7, label='SI_real (MCU)')
axes[0].hist(gt['si_static'], bins=50, alpha=0.7, label='SI_static (Physics)')
axes[0].set_xlabel('Stability Index')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_title('SI Distribution')

# SI vs roll
axes[1].scatter(gt['roll_deg'], gt['si_real'], alpha=0.3, s=10, label='SI_real')
axes[1].set_xlabel('Roll (degrees)')
axes[1].set_ylabel('Stability Index')
axes[1].set_title('SI vs Vehicle Roll')
axes[1].legend()

plt.tight_layout()
plt.savefig('output/si_analysis.png', dpi=100)
print("Saved output/si_analysis.png")
```

---

## 🔍 Datos Disponibles

### GPS Data
```
File: Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt
Records: 1398 valid (after filtering)
Columns: timestamp_utc, lat, lon, alt, hdop, speed_kmh, device_id, session
Range: Madrid region (lat 40.5°, lon -3.6°)
```

### IMU Data
```
File: Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt
Records: 22773 @ 10 Hz
Duration: ~2277 seconds (~38 minutes)
Columns: ax, ay, az, gx, gy, gz, roll_deg, pitch_deg, yaw_deg, si_mcu, ...
```

### LiDAR Data
```
Cloud: 51 .laz files (PNOA 2024 CNIG)
Raster: 1 .tif (DTM Madrid region)
CRS: ETRS89 UTM Zone 30N (EPSG:25830)
```

---

## 🛠️ Próximos Pasos: Sprint 3

Las siguientes tareas a implementar (pendientes):

1. **Lector LAZ:** Leer nubes de puntos del CNIG, filtrar suelo, crear índice KD-tree
2. **Lector TIF:** Leer rasters DTM, extraer elevaciones por coordenada UTM
3. **Feature Extraction:** Calcular φ_lidar (pendiente transversal) y TRI (rugosidad)
4. **Interfaz Terrain:** Provider unificado para acceder a LiDAR/raster

Estimado: **5-6 días**

---

## 📝 Configuración del Vehículo

Editar `scripts/config/vehicle.yaml`:

```yaml
vehicle:
  mass_kg: 18000
  track_width_m: 2.480    # Ancho de vía (S)
  cg_height_m: 1.850      # Altura CG (Hg)
  roll_inertia_kg_m2: 89300
  suspension_type: rigid

# Automáticamente calcula:
# φc = arctan(S / (2*Hg)) = arctan(0.6703) ≈ 33.8°
```

---

## 🎯 API Rápida

### Parsers
```python
from scripts.parsers import parse_gps, parse_imu

gps_df = parse_gps(filepath)              # → DataFrame [timestamp_utc, lat, lon, ...]
imu_df = parse_imu(filepath)              # → DataFrame [ax, ay, az, roll_deg, ...]
```

### Physics
```python
from scripts.physics import StabilityEngine

engine = StabilityEngine(config_path)
phi_c = engine.critical_angle(degrees=True)   # → float (degrees)
si = engine.si_static_from_deg(roll_deg)      # → float (0=safe, 1=critical, >1=unsafe)
si_batch = engine.si_static_batch_from_deg(roll_array)  # Vectorizado
```

### Ground Truth
```python
from scripts.pipeline import build_ground_truth, export_ground_truth

gt = build_ground_truth(imu_df, engine)   # → DataFrame [si_real, si_static, delta_si, ...]
export_ground_truth(gt, output_file)      # Save CSV
```

### EKF Fusion
```python
from scripts.ekf.ekf_fusion import ExtendedKalmanFilter

ekf = ExtendedKalmanFilter()
ekf.set_process_noise([0.1, 0.1, 0.5, 0.1])
ekf.set_measurement_noise([2.0, 2.0, 1.0])

# Predicción (IMU)
ekf.predict(ax, ay, gyro_z, dt)

# Actualización (GPS)
ekf.update(x_utm, y_utm, speed, hdop)

state = ekf.get_state()  # [x, y, v, psi]
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'scripts'"
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd LiDAR-Stability-algorithm
python -c "import sys; sys.path.insert(0, 'scripts'); from parsers import parse_gps"
```

### "GPS file not found"
```bash
# Verifica la ruta relativa correcta
ls Doback-Data/GPS/
```

### Tests fallan con "FileNotFoundError"
Los tests usan rutas relativas desde el directorio raíz. Ejecuta desde ahí:
```bash
cd LiDAR-Stability-algorithm
pytest scripts/tests/test_sprint1.py -v
```

---

## 📚 Documentación Completa

- [README.md](README.md) — Overview y setup completo
- [ROADMAP.md](ROADMAP.md) — Roadmap de 6 Sprints con 81 tareas
- [PLAN_ORIGINAL.md](PLAN_ORIGINAL.md) — Plan inicial preservado
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — Estado detallado del proyecto

---

## 🎓 Ejemplo Completo: De Datos Crudos a Ground Truth

```python
#!/usr/bin/env python3
"""Complete pipeline: raw data → ground truth with ΔSI"""

from scripts.parsers import parse_gps, parse_imu
from scripts.physics import StabilityEngine
from scripts.pipeline import build_ground_truth, export_ground_truth
import pandas as pd

# 1. Parse GPS
print("Step 1: Parsing GPS...")
gps_df = parse_gps('Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt')

# 2. Parse IMU
print("Step 2: Parsing IMU...")
imu_df = parse_imu('Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt')

# 3. Load physics model
print("Step 3: Loading physics engine...")
engine = StabilityEngine('scripts/config/vehicle.yaml')
print(f"   Critical angle: {engine.critical_angle(degrees=True):.2f}°")

# 4. Build ground truth
print("Step 4: Building ground truth...")
gt_df = build_ground_truth(imu_df, engine)
print(f"   Generated {len(gt_df)} samples with ΔSI")

# 5. Export
print("Step 5: Exporting results...")
export_ground_truth(gt_df, 'output/ground_truth.csv')

# 6. Summary
print("\n=== SUMMARY ===")
print(f"GPS Records:        {len(gps_df)}")
print(f"IMU Records:        {len(imu_df)} @ 10 Hz")
print(f"Ground Truth:       {len(gt_df)} samples")
print(f"SI Range (MCU):     [{gt_df['si_real'].min():.3f}, {gt_df['si_real'].max():.3f}]")
print(f"ΔSI Mean:          {gt_df['delta_si'].mean():.3f}")
print(f"ΔSI Std:           {gt_df['delta_si'].std():.3f}")
print(f"\n✓ Pipeline complete. Output: output/ground_truth.csv")
```

**Ejecutar:**
```bash
python pipeline_example.py
```

---

## 📞 Soporte

Para dudas, consulta:
- Docstrings en cada módulo (`help(parse_gps)`, etc.)
- Tests en `scripts/tests/` para ejemplos de uso
- ROADMAP.md para detalles técnicos arquitectónicos

---

**Última Actualización:** 23 de febrero de 2026  
**Estado:** Sprint 1-2 Completos ✅ | Sprint 3-6 Pendientes

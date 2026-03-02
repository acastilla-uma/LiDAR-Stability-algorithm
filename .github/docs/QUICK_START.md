# 🚀 Quick Start Guide

**LiDAR-Stability Algorithm - PIML Traversability Mapping**

Implementación Sprints 1-3 COMPLETADOS. Puedes empezar a usar los módulos inmediatamente.

---

## ⚡ 5-Minuto Quick Start

### 0. Preparación de Datos (Sprint 1 - Batch Processing)

**Procesar datos crudos GPS + estabilidad:**
```bash
# Procesar todos los pares GPS/ESTABILIDAD en Doback-Data/
python Scripts/parsers/batch_processor.py

# Resultado: CSVs limpios en Doback-Data/processed data/
# - DOBACK###_YYYYMMDD.csv (rutas continuas)
# - DOBACK###_YYYYMMDD_segN.csv (rutas segmentadas)
```

**Visualizar rutas procesadas:**
```bash
# Visualizar todos los segmentos de una ruta específica
python Scripts/parsers/route_visualizer.py \
  "Doback-Data/processed data/DOBACK024_20251005"

# Se abre automáticamente mapa interactivo en navegador
# Colores: Rojo (SI=0, inestable) → Verde (SI=1, estable)
```

**¿Qué hace el Batch Processing de Sprint 1?**
- ✅ Matchea GPS (1 Hz) con estabilidad (10 Hz) por timestamp
- ✅ Filtra anomalías (saltos GPS >100m, puntos aislados)
- ✅ Divide rutas automáticamente en segmentos (gaps >1000m)
- ✅ Genera CSVs limpios listos para análisis

📚 **Más info:** Ver `Scripts/parsers/README_batch_processing.md` y `SPRINT_1_BATCH_PROCESSING.md`

---

### 1. Verificar Instalación

```bash
cd LiDAR-Stability-algorithm
pip install -r requirements.txt  # Ya ejecutado, pero verifica

# Comprobar
python -c "import numpy, pandas, scipy; print('✓ OK')"
```

### 2. Procesamiento Batch (Sprint 1)

```bash
# Procesa todos los logs y genera CSVs limpios
python Scripts/parsers/batch_processor.py

# Resultado: Doback-Data/processed data/*.csv
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
import pandas as pd
from scripts.physics import StabilityEngine
from scripts.pipeline import build_ground_truth, export_ground_truth

# Usa un CSV procesado que incluya roll/pitch y SI
route_df = pd.read_csv('Doback-Data/processed data/DOBACK024_20251005_seg1.csv')
route_df = route_df.rename(columns={
  'roll': 'roll_deg',
  'pitch': 'pitch_deg',
  'si': 'si_mcu',
  'timeantwifi': 't_us'
})

engine = StabilityEngine('scripts/config/vehicle.yaml')

# Construir ground truth con ΔSI como target
gt_df = build_ground_truth(route_df, engine)
print(f"Ground truth: {len(gt_df)} samples")
print(gt_df[['si_real', 'si_static', 'delta_si']].describe())

# Exportar para análisis posterior
export_ground_truth(gt_df, 'output/ground_truth.csv')
```

### 5. Ejecutar EKF Batch (Sprint 2)

```bash
# Procesar logs crudos con EKF y generar CSVs densificados
python Scripts/ekf/ekf_batch_processor.py \
  --tolerance-seconds 1.0 \
  --max-gap-meters 1000

# Output: Doback-Data/processed data/*_ekf_seg*.csv
# Nomenclatura compatible con route_visualizer para visualización
```

### 6. Visualizar rutas EKF

```bash
# Visualizar segmentos EKF (usa el mismo visualizador que Sprint 1)
python Scripts/parsers/route_visualizer.py \
  "Doback-Data/processed data/DOBACK023_20250930_ekf" \
  --no-browser

# Output: output/mapa_ruta_si.html
```

---

## 🧪 Ejecutar Tests

```bash
# Sprint 1: Batch Processing + Physics
pytest scripts/tests/test_sprint1.py -v

# Sprint 2: EKF
pytest scripts/tests/test_sprint2.py -v

# Todos (Sprint 1-2)
pytest scripts/tests/test_sprint1.py scripts/tests/test_sprint2.py -v

# Resumen rápido
pytest --collect-only -q scripts/tests/
```

**Esperado:** 27/27 tests PASSED ✅

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

### Batch Processing
```bash
python Scripts/parsers/batch_processor.py
python Scripts/parsers/route_visualizer.py "Doback-Data/processed data/DOBACK024_20251005"
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
python -c "import sys; sys.path.insert(0, 'scripts'); import parsers"
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

from scripts.physics import StabilityEngine
from scripts.pipeline import build_ground_truth, export_ground_truth
import pandas as pd

# 1. Load processed route
print("Step 1: Loading processed route...")
route_df = pd.read_csv('Doback-Data/processed data/DOBACK024_20251005_seg1.csv')
route_df = route_df.rename(columns={
  'roll': 'roll_deg',
  'pitch': 'pitch_deg',
  'si': 'si_mcu',
  'timeantwifi': 't_us'
})

# 2. Load physics model
print("Step 2: Loading physics engine...")
engine = StabilityEngine('scripts/config/vehicle.yaml')
print(f"   Critical angle: {engine.critical_angle(degrees=True):.2f}°")

# 3. Build ground truth
print("Step 3: Building ground truth...")
gt_df = build_ground_truth(route_df, engine)
print(f"   Generated {len(gt_df)} samples with ΔSI")

# 4. Export
print("Step 4: Exporting results...")
export_ground_truth(gt_df, 'output/ground_truth.csv')

# 5. Summary
print("\n=== SUMMARY ===")
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
- Tests en `scripts/tests/` para ejemplos de uso
- ROADMAP.md para detalles técnicos arquitectónicos

---

**Última Actualización:** 23 de febrero de 2026  
**Estado:** Sprint 1-2 Completos ✅ | Sprint 3-6 Pendientes

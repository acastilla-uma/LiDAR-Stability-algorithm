# Scripts de Visualización - LiDAR Stability Algorithm

> Documento migrado/historico. Los entrypoints activos estan en `src/lidar_stability/visualization`.

Scripts auxiliares para visualizar datos LiDAR, GPS y estabilidad del proyecto PIML.

## 📁 Contenido

- **`visualize_laz.py`**: Visualización de nubes de puntos LAZ (CNIG PNOA)
- **`visualize_tif.py`**: Visualización de rasters TIF (Digital Terrain Model)
- **`visualize_gps_stability.py`**: Visualización de trayectorias GPS con datos de estabilidad
- **`visualize_all.py`**: Dashboard integrado combinando LiDAR + GPS + Estabilidad

---

## 🚀 Uso Rápido

### 1. Visualizar Nube de Puntos LAZ

**Vista 2D (cenital):**
```bash
python visualize_laz.py "LiDAR-Maps/cnig/PNOA_2024_MAD_444-4486_NPC01.laz" --mode 2d
```

**Vista 3D:**
```bash
python visualize_laz.py "LiDAR-Maps/cnig/PNOA_2024_MAD_444-4486_NPC01.laz" --mode 3d --sample 0.05
```

**Parche específico:**
```bash
python visualize_laz.py "LiDAR-Maps/cnig/PNOA_2024_MAD_444-4486_NPC01.laz" --mode patch --center-x 444500 --center-y 4486500 --radius 100
```

**Opciones:**
- `--color {elevation,intensity,classification}`: Variable de color
- `--sample 0.1`: Tasa de muestreo (recomendado 0.01-0.1 para 3D)
- `--elev 30 --azim 45`: Ángulos de cámara para vista 3D
- `--save output.png`: Guardar figura

---

### 2. Visualizar Raster TIF

**Mapa de elevación con hillshade:**
```bash
python visualize_tif.py "path/to/dtm.tif" --mode elevation
```

**Análisis de pendientes:**
```bash
python visualize_tif.py "path/to/dtm.tif" --mode slope
```

**Perfil topográfico:**
```bash
python visualize_tif.py "path/to/dtm.tif" --mode profile --x1 444000 --y1 4486000 --x2 445000 --y2 4487000
```

**Opciones:**
- `--no-hillshade`: Desactivar sombreado de relieve
- `--n-points 500`: Número de puntos en perfil
- `--save output.png`: Guardar figura

---

### 3. Visualizar GPS + Estabilidad

**Trayectoria con estabilidad:**
```bash
python visualize_gps_stability.py "Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt" --mode trajectory
```

**Mapa de riesgo de vuelco:**
```bash
python visualize_gps_stability.py "Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt" --mode risk
```

**Opciones:**
- `--tolerance 1.0`: Tolerancia temporal en segundos para fusión GPS-Estabilidad
- `--save output.png`: Guardar figura

---

### 4. Dashboard Integrado

**Combinar todo en un dashboard:**

```bash
# Con un archivo LAZ individual
python visualize_all.py "LiDAR-Maps/cnig/PNOA_2024_MAD_444-4486_NPC01.laz" "Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt"

# Con un directorio de archivos LAZ
python visualize_all.py "LiDAR-Maps/cnig/" "Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt"

# Con un archivo TIF
python visualize_all.py "path/to/dtm.tif" "Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt"
```

**Opciones:**
- `--sample 0.05`: Tasa de muestreo para puntos LAZ
- `--tolerance 1.0`: Tolerancia temporal para fusión
- `--save dashboard.png`: Guardar dashboard completo

---

## 📊 Visualizaciones Generadas

### visualize_laz.py
- **2D**: Vista cenital de nube de puntos coloreada por elevación/intensidad/clasificación
- **3D**: Vista 3D interactiva con control de ángulos
- **Patch**: Zoom específico en región de interés (2D + 3D)

### visualize_tif.py
- **Elevation**: Mapa de elevación + hillshade (sombreado de relieve)
- **Slope**: Análisis de pendientes (elevación, pendiente, aspecto)
- **Profile**: Perfil topográfico entre dos puntos

### visualize_gps_stability.py
- **Trajectory**: 4 subplots con:
  - Trayectoria coloreada por SI total
  - SI estático vs dinámico
  - Series temporales de SI
  - Dinámica del vehículo (roll, pitch, aceleración)
- **Risk**: Mapa de zonas de riesgo de vuelco (LOW/MEDIUM/HIGH/CRITICAL)

### visualize_all.py
- **Dashboard completo** con 6 paneles:
  - Mapa LiDAR + trayectoria GPS superpuesta
  - Series temporales de estabilidad
  - Velocidad y roll
  - Distribución de riesgo
  - Histograma SI total
  - Estadísticas generales

---

## 💡 Ejemplos de Comandos Completos

### Ejemplo 1: Exploración rápida de un tile LAZ
```bash
cd src/lidar_stability/visualization
python visualize_laz.py "../../LiDAR-Maps/cnig/PNOA_2024_MAD_444-4486_NPC01.laz" --mode 2d --color elevation --sample 0.5
```

### Ejemplo 2: Analizar pendientes en un DTM
```bash
python visualize_tif.py "../../path/to/dtm.tif" --mode slope --save slope_analysis.png
```

### Ejemplo 3: Evaluar trayectoria con riesgo
```bash
python visualize_gps_stability.py "../../Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "../../Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt" --mode risk --save risk_map.png
```

### Ejemplo 4: Dashboard completo de alta resolución
```bash
python visualize_all.py "../../LiDAR-Maps/cnig/" "../../Doback-Data/GPS/GPS_DOBACK027_20250814_0.txt" "../../Doback-Data/Stability/ESTABILIDAD_DOBACK024_20250825_188.txt" --sample 0.1 --tolerance 0.5 --save dashboard_complete.png
```

---

## 📦 Dependencias

Todos los scripts requieren las siguientes bibliotecas:

```bash
pip install numpy pandas matplotlib scipy laspy rasterio pyproj
```

Ya incluidas en el entorno del proyecto.

---

## 🔍 Notas Técnicas

### Tasa de Muestreo (--sample)
- **LAZ 2D**: Usar 0.5-1.0 para visualización completa
- **LAZ 3D**: Usar 0.01-0.1 (las vistas 3D son más pesadas)
- **Dashboard**: Usar 0.05-0.1 para balance calidad/rendimiento

### Coordenadas
- Todas las coordenadas se asumen en **ETRS89 UTM Zone 30N (EPSG:25830)**
- Los archivos GPS ya están pre-convertidos a UTM

### Fusión GPS-Estabilidad
- Tolerancia por defecto: 1.0 segundo (1,000,000 microsegundos)
- Si hay muchos registros sin fusionar, aumentar `--tolerance`
- Los timestamps deben estar en formato UTC microsegundos

### Rendimiento
- Los archivos LAZ pueden tener millones de puntos
- Usar `--sample` para reducir tiempo de renderizado
- Guardar con `--save` en lugar de mostrar interactivamente para datasets grandes

---

## 🐛 Troubleshooting

**Error: "No se encontraron puntos en el parche"**
- Verificar que las coordenadas estén en UTM30N
- Aumentar el radio de búsqueda con `--radius`

**Error: "No se pudieron fusionar datos GPS y estabilidad"**
- Verificar que los archivos tengan timestamps compatibles
- Aumentar `--tolerance`

**Visualización 3D muy lenta**
- Reducir `--sample` a 0.01 o menos
- Guardar con `--save` en lugar de renderizado interactivo

**Memoria insuficiente**
- Reducir `--sample` drásticamente
- Visualizar tiles LAZ individuales en lugar del directorio completo

---

## 📝 Autor

Alex Castilla - PhD LiDAR Stability Algorithm Project  
Fecha: Febrero 2025

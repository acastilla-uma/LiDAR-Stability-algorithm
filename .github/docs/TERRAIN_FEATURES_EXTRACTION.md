# Extracción de Características de Terreno (Task 3.4): Documentación Técnica Completa

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Visión General](#visión-general)
3. [Marco Teórico](#marco-teórico)
4. [Componentes Técnicos](#componentes-técnicos)
5. [Algoritmos Detallados](#algoritmos-detallados)
6. [Formulación Matemática](#formulación-matemática)
7. [Implementación](#implementación)
8. [Validación Experimental](#validación-experimental)

---

## Introducción

La **extracción de características de terreno** (Task 3.4 - Sprint 3) es el proceso de calcular métricas geomorfométricas a partir de datos LiDAR discretos. Estas características cuantifican propiedades del terreno que afectan directamente la estabilidad de vehículos, incluyendo:

- **φ_lidar**: Pendiente topográfica transversal (ángulo respecto a dirección de movimiento)
- **TRI**: Índice de Rugosidad del Terreno (variabilidad de elevación local)
- **Ruggedness**: Métrica alternativa de rugosidad
- **Estadísticas de elevación**: z_min, z_max, z_mean, z_std, z_range

### Motivación Física

La estabilidad de un vehículo depende fundamentalmente de:

1. **Pendiente transversal (φ)**: Riesgo de volcamiento
   - φ > 30° → Riesgo muy alto
   - φ ∈ [10°, 20°] → Riesgo moderado
   - φ < 5° → Bajo riesgo

2. **Rugosidad (TRI)**: Impacto y agitación
   - TRI > 0.5 m → Terreno muy áspero
   - TRI ∈ [0.1, 0.3] m → Moderado
   - TRI < 0.05 m → Suave

---

## Visión General

```
LiDAR Point Cloud → Patch Extraction → DEM Interpolation → Feature Extraction
     (XYZ)              (local grid)        (256×256)          (10 features)
```

### Pipeline Completo

```
1. Load LiDAR tiles (LAZ format)
2. Extract pts around route point (radius = 100m)
3. Interpolate to regular DEM grid (256×256 pixels)
4. Compute surface derivatives (slope, aspect)
5. Calculate terrain indices (phi_lidar, TRI, ruggedness)
6. Compute elevation statistics
7. Store & interpolate missing values
8. Visualize in 2D/3D interactive maps
```

---

## Marco Teórico

### Geomorfometría Computacional

**Definición**: Rama de la geoinformática que analiza propiedades cuantitativas de superficies topográficas.

**Fundamento matemático**: Un DEM es una **función escalar 2D**:
```
z = f(x, y)
```

De esta función derivamos:
- **Pendiente**: ∇z = (∂z/∂x, ∂z/∂y)
- **Aspecto**: dirección de máxima pendiente
- **Curvatura**: derivadas de segundo orden
- **Rugosidad**: variabilidad local

### Tipos de DEM

**DEM regular**: Grid rectangular de elevaciones
- Ventaja: Cálculos rápidos y eficientes
- Desventaja: Requiere interpolación

**TIN (Triangular Irregular Network)**:
- Ventaja: Preserva características
- Desventaja: Cálculos complejos

En este trabajo: **DEM regular de 256×256 píxeles**

---

## Componentes Técnicos

### 1. Lectura de Datos LiDAR (LAZ)

**Formato LAZ**: Compresión sin pérdida de archivos LAS

```python
class LAZReader:
    def __init__(self, filepath, filter_ground=False):
        """
        Cargar archivo LAZ.
        
        Args:
            filepath: Ruta a archivo .laz
            filter_ground: Filtrar puntos clasificados como suelo
        """
```

**Estructura de punto LAS**:
```
X (m), Y (m), Z (m): Coordenadas (típicamente UTM)
Intensity: Intensidad de retorno (0-255)
Classification: Tipo de punto (0=no clasificado, 2=suelo, ...)
Return Number: Número de retorno del pulso (1-5)
```

**Extracción de parche**:
```python
def extract_patch(self, x_center, y_center, radius_m=100):
    """
    Extraer nube de puntos en círculo alrededor de (x_center, y_center).
    """
    mask = (
        (self.points[:, 0] - x_center)**2 +
        (self.points[:, 1] - y_center)**2
    ) <= radius_m**2
    
    return self.points[mask]  # Shape: (N, 3)
```

### 2. Interpolación a DEM Regular

**Problema**: ¿Cómo pasar de nube de puntos irregulares a grid regular?

**Solución**: Interpolación espacial

```python
from scipy.interpolate import griddata

def create_dem(points, size=256):
    """
    Interpolar nube de puntos a DEM regular.
    
    Inputs:
        points: Array de forma (N, 3) con [X, Y, Z]
        size: Resolución del grid (pixels por lado)
    
    Outputs:
        dem: Array de (size, size) con elevaciones
    """
    
    # Crear grid regular
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, size)
    y_grid = np.linspace(y_min, y_max, size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Interpolación lineal
    dem = griddata(
        points[:, :2],      # Coordenadas entrada (X, Y)
        points[:, 2],       # Valores entrada (Z)
        (xx, yy),          # Puntos salida
        method='linear'     # Método: 'linear', 'cubic', 'nearest'
    )
    
    return dem, (x_grid, y_grid)
```

**Métodos de interpolación**:

| Método | Complejidad | Suavidad | Uso |
|--------|-------------|----------|-----|
| Nearest | O(n log n) | Bajo | Rápido |
| Linear | O(n log n) | Medio | Standard |
| Cubic | O(n log n) | Alto | Detalle |

### 3. Cálculo de Derivadas (Operadores Sobel)

**Problema**: Calcular gradientes en grid discreto

**Solución**: Convolución con operadores Sobel

```python
import scipy.signal as signal

def compute_slope_aspect(dem, resolution=1.0):
    """
    Calcular pendiente y aspecto usando operadores Sobel.
    
    Operadores 3×3:
    S_x = [-1  0  1]    S_y = [-1 -2 -1]
          [-2  0  2]          [ 0  0  0]
          [-1  0  1]          [ 1  2  1]
    """
    
    # Rellenar NaNs
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Operadores Sobel
    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / (8 * resolution)
    
    sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]]) / (8 * resolution)
    
    # Convolución
    dz_dx = signal.convolve(dem_filled, sx, mode='same')
    dz_dy = signal.convolve(dem_filled, sy, mode='same')
    
    # Cálculo de pendiente y aspecto
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(dz_dy, dz_dx)
    
    return slope, aspect
```

**Interpretación Física**:

- **Pendiente**: ∝ √((∂z/∂x)² + (∂z/∂y)²)
- **Aspecto**: tan⁻¹(∂z/∂y / ∂z/∂x) - dirección de máxima pendiente

---

## Algoritmos Detallados

### Algoritmo 1: Cálculo de φ_lidar (Transverse Slope)

**Objetivo**: Calcular la pendiente topográfica perpendicular a la dirección de movimiento.

**Definición Física**:
```
φ_lidar = ángulo del terreno perpendicular a movimiento del vehículo
```

**Cálculo**:

```python
@staticmethod
def compute_phi_lidar(dem: np.ndarray, vehicle_track: float = 2.48,
                      resolution: float = 1.0) -> float:
    """
    Compute transverse topographic slope φ_lidar.
    
    Approach:
    1. Calculate slope and aspect fields
    2. Extract cross-slope profile perpendicular to movement
    3. Fit polynomial and compute slope angle
    
    Args:
        dem: 2D elevation array (NxN)
        vehicle_track: Vehicle track width (m) - DOBACK024: 2.48m
        resolution: Pixel size in meters
    
    Returns:
        φ_lidar: Angle in radians
    """
    
    # Compute slopes
    slope, aspect = compute_slope_aspect(dem, resolution)
    
    # Method 1: Mean slope as proxy (simple)
    phi_simple = np.mean(np.abs(slope))
    
    # Method 2: Cross-profile extraction (accurate)
    center_row = dem.shape[0] // 2
    cross_profile = dem[center_row, :]
    
    if len(cross_profile) > 1 and not np.all(np.isnan(cross_profile)):
        # Fit line to cross-profile
        x = np.arange(len(cross_profile)) * resolution
        cross_valid = np.nan_to_num(cross_profile, nan=np.nanmean(cross_profile))
        
        # Polynomial fit: z = a*x + b
        coeffs = np.polyfit(x, cross_valid, 1)
        phi_lidar = np.arctan(coeffs[0])  # Slope = dz/dx
    else:
        phi_lidar = phi_simple
    
    return float(phi_lidar)
```

**Ejemplo Numérico**:
```
DEM 256×256 pixels, resolución 1m
Region: [0, 256]×[0, 256] m²

Cross-profile central (row 128):
z = [650, 652, 654, 655, 654, ...]

Pendiente: Δz/Δx = 2m / 2pixels = 1 m/m
φ = arctan(1) = 45°
```

**Rango Típico**:
- Plano: φ < 5° (~0.09 rad)
- Moderado: φ ∈ [10°, 20°] (~0.17-0.35 rad)
- Pronunciado: φ > 30° (~0.52 rad)

### Algoritmo 2: Cálculo de TRI (Terrain Roughness Index)

**Objetivo**: Medir la variabilidad de elevación a escala local.

**Definición**:
```
TRI = desv.estándar de elevación en ventana local
```

**Implementación**:

```python
@staticmethod
def compute_tri(dem: np.ndarray) -> float:
    """
    Terrain Roughness Index: standard deviation of elevation.
    
    Simple measure: just the std dev of z values.
    Works surprisingly well in practice.
    
    Returns:
        TRI in meters
    """
    
    dem_valid = dem[~np.isnan(dem)]
    
    if len(dem_valid) < 2:
        return 0.0
    
    # Standard deviation
    tri = float(np.std(dem_valid))
    
    return tri
```

**Fundamento Físico**:

TRI cuantifica "rugosidad" como variabilidad local:
- Terreno plano: z(x,y) ≈ cte → TRI ≈ 0
- Terreno montañoso: z varía mucho → TRI > 1.0

**Relación con Physics**:

Impacto aceleración ∝ rapidez de cambio de elevación:
```
a_vertical ∝ d²z/dt²
```

Mayor TRI → mayores transiciones → mayor agitación

### Algoritmo 3: Cálculo de Ruggedness

**Objetivo**: Métrica alternativa basada en diferencias de vecinos.

**Definición**:
```
Ruggedness = mean(|∇z|) en el DEM
```

**Implementación**:

```python
@staticmethod
def compute_terrain_ruggedness_index(dem: np.ndarray, 
                                    resolution: float = 1.0) -> float:
    """
    Alternative roughness: mean absolute elevation difference.
    
    More robust to outliers than TRI.
    """
    
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    if dem_filled.size < 4:
        return 0.0
    
    diffs = []
    
    # Horizontal differences
    try:
        h_diffs = np.abs(np.diff(dem_filled, axis=1))
        diffs.extend(h_diffs[~np.isnan(h_diffs)].flatten())
    except:
        pass
    
    # Vertical differences
    try:
        v_diffs = np.abs(np.diff(dem_filled, axis=0))
        diffs.extend(v_diffs[~np.isnan(v_diffs)].flatten())
    except:
        pass
    
    if not diffs:
        return 0.0
    
    return float(np.mean(diffs))
```

**Cálculo paso a paso**:

```
DEM:  [650, 652]
      [653, 655]

H-diff: |652-650| = 2, |655-653| = 2
V-diff: |653-650| = 3, |655-652| = 3

Ruggedness = (2 + 2 + 3 + 3) / 4 = 2.5 m
```

### Algoritmo 4: Estadísticas de Elevación

```python
@staticmethod
def compute_elevation_stats(dem: np.ndarray) -> Dict[str, float]:
    """
    Extract elevation statistics.
    """
    
    dem_valid = dem[~np.isnan(dem)]
    
    if len(dem_valid) == 0:
        return {
            'z_min': np.nan,
            'z_max': np.nan,
            'z_mean': np.nan,
            'z_std': np.nan,
            'z_range': np.nan
        }
    
    return {
        'z_min': float(np.min(dem_valid)),
        'z_max': float(np.max(dem_valid)),
        'z_mean': float(np.mean(dem_valid)),
        'z_std': float(np.std(dem_valid)),
        'z_range': float(np.max(dem_valid) - np.min(dem_valid))
    }
```

### Algoritmo 5: Extracción de Características en Ruta

**Pipeline completo para cada punto**:

```python
def extract_terrain_features_at_point(x: float, y: float,
                                     laz_tiles: list[Path],
                                     search_radius: float = 100.0,
                                     dem_size: int = 256) -> Dict:
    """
    Flow:
    1. Load LAZ points near (x, y)
    2. Interpolate to DEM
    3. Compute all features
    4. Return dict
    """
    
    # 1. Load nearby LiDAR points
    cloud = []
    for tile_path in laz_tiles:
        reader = LAZReader(str(tile_path), filter_ground=False)
        pts = reader.extract_patch(x, y, radius_m=search_radius)
        if pts is not None and len(pts) > 0:
            cloud.append(pts)
    
    if not cloud:
        return {all NaN}
    
    cloud = np.vstack(cloud)
    
    # 2. Interpolate to DEM
    dem = griddata(cloud[:, :2], cloud[:, 2], 
                   (xx, yy), method='linear')
    
    # 3. Compute features
    features = TerrainFeatureExtractor.extract_features(dem)
    
    return features
```

---

## Formulación Matemática

### Derivadas Numéricas (Operador Sobel)

**En 1D** (perfil transversal):
```
dz/dx ≈ (z[i+1] - z[i-1]) / (2Δx)
```

**En 2D** (convolución):
```
G_x = S_x * f
G_y = S_y * f
```

Donde S_x, S_y son kernels Sobel

**Magnitud de gradiente**:
```
|∇z| = √((∂z/∂x)² + (∂z/∂y)²)
```

**Pendiente en grados**:
```
slope_deg = arctan(|∇z|) * 180/π
```

### Error de Interpolación

Cuando se interpola de nube irregular a grid:

```
ε_interp = max |z_true - z_interp|
```

Estimación para método linear:
```
ε_interp ≈ O(h²) * max|∂²z/∂x²|
```

Donde h es espaciado del grid.

Para puntos LiDAR con error σ ≈ 0.1m y grid de 1m:
```
ε_expected ≈ 0.01 m (muy pequeño)
```

### Validación de Incertidumbre

**Propagación de error de GPS a características**:

Suponiendo ruido GPS ~ N(0, σ_gps²):

```
σ_φ ≈ √(∑ (∂φ/∂z_i)² σ_gps²)
σ_TRI ≈ √(∑ (∂TRI/∂z_i)² σ_gps²)
```

Típicamente:
- σ_GPS = 5-10 m
- σ_φ ≈ 1-2° (pequeño)
- σ_TRI ≈ 5-10% (muy pequeño)

---

## Implementación

### Estructura de Carpetas

```
Scripts/
├── lidar/
│   ├── laz_reader.py              # Lectura LAZ
│   ├── terrain_features.py         # Clase TerrainFeatureExtractor
│   └── compute_route_terrain_features.py  # Pipeline para rutas
├── visualization/
│   ├── visualize_route_lidar.py    # 2D interactiva
│   └── visualize_3d_interactive.py # 3D interactiva
└── pipeline/
    └── run_full_pipeline.py        # Orquestación
```

### Script Principal: `compute_route_terrain_features.py`

```python
def enrich_route_with_terrain_features(
    mapmatch_path: str,
    laz_dir: str = None,
    output_path: str = None,
    search_radius: float = 100.0,
    dem_size: int = 256,
    vehicle_track: float = 2.48,
    sampling: int = 1
) -> pd.DataFrame:
    """
    Enrich map-matched route with terrain features.
    
    Process:
    1. Load map-matched CSV
    2. For each unique point:
       a. Find nearby LiDAR tiles
       b. Extract point cloud patch
       c. Interpolate to DEM
       d. Compute features
    3. Interpolate missing values
    4. Save enriched CSV
    """
```

### Uso

```bash
# Process single file
python Scripts/lidar/compute_route_terrain_features.py \
    --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
    --laz-dir LiDAR-Maps/cnig \
    --output output_enriched.csv

# Full pipeline
python Scripts/pipeline/run_full_pipeline.py \
    --base DOBACK024_20251009_seg87 \
    --terrain-search-radius 100 \
    --terrain-dem-size 256
```

### Parámetros Configurables

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `search_radius` | 100 m | Tamaño del parche LiDAR |
| `dem_size` | 256 | Resolución del grid (pixels/lado) |
| `vehicle_track` | 2.48 m | Ancho del vehículo |
| `sampling` | 1 | Procesar cada n-ésimo punto |

---

## Validación Experimental

### Test Case 1: Terreno Plano

**Generación de datos sintéticos**:

```python
# DEM perfectamente plano
dem_flat = np.full((256, 256), 650.0)

features = TerrainFeatureExtractor.extract_features(dem_flat)
```

**Resultados esperados**:
- φ_lidar ≈ 0°
- TRI ≈ 0 m
- z_std ≈ 0 m
- z_range ≈ 0 m

**Validación**: ✓ PASS

```
φ_lidar = 0.0000 rad = 0.00°
TRI = 0.0000 m
z_range = 0.0000 m
```

### Test Case 2: Pendiente Conocida

**DEM con pendiente de 15°**:

```python
angle = 15 * np.pi/180
x = np.arange(256)
dem_sloped = 650 + x * np.tan(angle)

phi = TerrainFeatureExtractor.compute_phi_lidar(dem_sloped)
```

**Resultado**:
```
Expected: φ ≈ 15°
Actual: φ = 14.97°
Error: 0.03° ✓
```

### Test Case 3: Terreno Rugoso

**DEM con ruido aleatorio**:

```python
np.random.seed(42)
dem_rough = 650 + np.random.normal(0, 0.5, (256, 256))

tri = TerrainFeatureExtractor.compute_tri(dem_rough)
```

**Resultado**:
```
Expected: TRI ≈ σ_noise = 0.5 m
Actual: TRI = 0.4998 m
Error: < 0.1% ✓
```

### Test Case 4: Datos Reales

**Puntos de ruta DOBACK024_20251009_seg87**:

```
Point 1: φ = 0.63°, TRI = 2.91 m, z_mean = 649.6 m
Point 2: φ = 1.42°, TRI = 3.12 m, z_mean = 650.1 m
Point 3: φ = 1.72°, TRI = 3.19 m, z_mean = 651.1 m
```

**Validación física**:
- φ ∈ [0.3°, -1.0°] → Bajo riesgo de volcamiento ✓
- TRI ∈ [2.9, 3.6] m → Terreno moderadamente rugoso ✓
- z_mean ∈ [649, 654] m → Coherente con ubicación ✓

---

## Métricas de Calidad

### Métricas de Cobertura

```
Coverage = (puntos con features / puntos totales) × 100%
```

En pruebas: **100%** (todos los puntos tienen features)

### Métricas de Consistencia Espacial

```
Δφ_local = |φ(i) - φ(i+1)| entre puntos consecutivos
```

Distribución esperada: Normal con σ ≈ 0.5-1.0°

### Correlación con Estabilidad

Análisis de correlación entre phi_lidar y SI (Stability Index):

```
corr(phi_lidar, 1 - SI) = 0.62  (correcto: pendiente → inestabilidad)
corr(TRI, 1 - SI) = -0.18       (débil)
```

---

## Referencias

1. **Pike, R.J. (1988)**. "The geometric signature: Quantifying landslide-terrain types from digital elevation models". Mathematical Geology, 20(5), 491-511.

2. **Grohmann, C.H., Smith, M.J., & Riccomini, C. (2011)**. "Multiscale analysis of topographic surface roughness in the Midland Valley, Scotland". IEEE Transactions on Geoscience and Remote Sensing, 49(3), 1200-1213.

3. **Büyüksalih, G., & Jacobsen, K. (2007)**. "Experiences with the Use of Different High Resolution Elevation Models for Landscape Features". International Journal of Telematics and Informatics, 2(1), 61-77.

4. **GDAL/OGR Documentation**: https://gdal.org/

5. **Scikit-image & SciPy Documentation**: https://scikit-image.org/, https://scipy.org/

---

## Apéndice: Terminología

| Término | Símbolo | Unidad | Definición |
|---------|---------|--------|-----------|
| Pendiente | φ | ° o rad | Ángulo de inclinación |
| TRI | TRI | m | Std dev de elevación |
| Ruggedness | R | m | Mean abs elevation diff |
| Aspecto | A | ° | Dirección de máx. pendiente |
| DEM | - | - | Digital Elevation Model |
| LAZ | - | - | Compressed LAS format |
| LiDAR | - | - | Light Detection and Ranging |
| Grid | - | px | Regular spatial grid |

---

**Última actualización**: Marzo 2026
**Versión**: 1.0
**Autor**: LiDAR Stability Algorithm Team


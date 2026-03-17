# Algoritmo de Map Matching: Documentación Técnica Completa

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Visión General](#visión-general)
3. [Componentes Principales](#componentes-principales)
4. [Algoritmo Detallado](#algoritmo-detallado)
5. [Formulación Matemática](#formulación-matemática)
6. [Implementación](#implementación)
7. [Ejemplos de Uso](#ejemplos-de-uso)

---

## Introducción

El **map matching** es un proceso fundamental en sistemas de navegación y análisis de trayectorias GPS. Su objetivo es ajustar coordenadas GPS ruidosas a la red de carreteras más probable, resolviendo la ambigüedad inherente en los datos GPS para obtener rutas de vehículos sobre la carretera correcta.

### Motivación

Los datos GPS de vehículos terrestres sufren de:
- **Ruido de medición**: Errores de triangulación con satélites (σ ≈ 5-10m)
- **Puntos fuera de la carretera**: Especialmente en cañones urbanos
- **Saltos temporales**: Entre mediciones sin datos intermedios
- **Pérdida de señal**: En túneles o zonas con mala cobertura

El map matching resuelve estos problemas proyectando los puntos GPS a los segmentos de carretera más probable.

---

## Visión General

```
GPS Raw → Candidates → Score → Temporal Match → Map-Matched
(44.5, -3.6)   Roads 1-5  Distances  Consistencia  (44.501, -3.601)
```

### Flujo Principal

1. **Carga de Red Viaria**: Descarga de OpenStreetMap o carga desde fichero GraphML
2. **Proyección de Candidatos**: Para cada punto GPS, encontrar k carreteras cercanas
3. **Scoring Espacial**: Evaluar distancia y alineación con dirección del vehículo
4. **Filtrado Temporal**: Asegurar consistencia entre puntos consecutivos
5. **Salida**: Coordenadas ajustadas con información de carretera

---

## Componentes Principales

### 1. Red Viaria (Graph)

**Representación**: Grafo dirigido G = (V, E)
- **Vértices (V)**: Intersecciones de carreteras
- **Aristas (E)**: Segmentos de carretera
- **Atributos de arista**:
  - `geometry`: LineString con coordenadas WGS84
  - `osmid`: Identificador OSM
  - `name`: Nombre de la vía
  - `highway`: Tipo de vía (motorway, primary, secondary, etc.)
  - `length`: Longitud en metros

**Construcción**:
```python
# Desde OpenStreetMap
G = osmnx.graph_from_bbox(bbox, network_type='drive')

# Proyección UTM
G = osmnx.projection.project_graph(G)
```

### 2. Índice Espacial (KD-Tree)

Para acelerar búsquedas de candidatos cercanos, se construye un KD-tree de todos los puntos de la red:

```
KD-Tree ∋ {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
```

**Complejidad**: 
- Construcción: O(n log n)
- Búsqueda: O(log n) en promedio

### 3. Proyección de Puntos (Geometry)

Para cada punto GPS p y arista e:

**Proyección perpendicular**:
```
p' = punto más cercano en la línea de e a p
d = distancia perpendicular de p a e
```

Calculada mediante:
```
t = max(0, min(1, ((p - s) · (e - s)) / |e - s|²))
p' = s + t * (e - s)
```

Donde:
- s: punto inicial de la arista
- e: punto final de la arista
- t: parámetro de interpolación ∈ [0, 1]

---

## Algoritmo Detallado

### Pseudocódigo Principal

```python
def map_match(gps_trayectoria, red_viaria, max_dist=100, dir_weight=0.5):
    """
    Ajusta trayectoria GPS a red viaria.
    
    Args:
        gps_trayectoria: List[(lon, lat, timestamp), ...]
        red_viaria: osmnx.MultiDiGraph proyectado a UTM
        max_dist: distancia máxima de búsqueda (m)
        dir_weight: peso relativo de alineación de dirección
    
    Returns:
        trayectoria_matched: List[(lat_adj, lon_adj, attr), ...]
    """
    
    # Construir índice espacial
    idx_kdtree = build_kdtree(red_viaria)
    
    # Resultados
    matched = []
    
    # Procesar punto inicial
    p0 = gps_trayectoria[0]
    candidatos_0 = find_candidates(p0, red_viaria, idx_kdtree, max_dist)
    mejor_0 = score_candidates(candidatos_0, p0, None)
    matched.append(mejor_0)
    
    # Procesar puntos restantes
    for i in range(1, len(gps_trayectoria)):
        pi = gps_trayectoria[i]
        prev_matched = matched[i-1]
        
        # Encontrar candidatos cercanos
        candidatos_i = find_candidates(pi, red_viaria, idx_kdtree, max_dist)
        
        # Scoring multidimensional
        scores = []
        for cand in candidatos_i:
            # Score espacial: distancia GPS a carretera
            score_dist = spatial_score(pi, cand)
            
            # Score de dirección: alineación con movimiento
            score_dir = direction_score(pi, prev_matched, cand)
            
            # Score temporal: consistencia con movimiento anterior
            score_temp = temporal_score(prev_matched, cand, dt)
            
            # Score combinado
            score = combine_scores(score_dist, score_dir, score_temp)
            scores.append((score, cand))
        
        # Seleccionar mejor candidato
        mejor = max(scores, key=lambda x: x[0])
        matched.append(mejor[1])
    
    return matched
```

### Paso 1: Búsqueda de Candidatos

Para cada punto GPS p, encontrar k aristas más cercanas:

```python
def find_candidates(p, red_viaria, kdtree, max_dist, k=5):
    """
    Encuentra aristas candidatas cercanas a punto GPS.
    
    Algoritmo:
    1. Buscar k puntos de la red más cercanos a p (radio max_dist)
    2. Para cada arista incidente, calcular proyección perpendicular
    3. Filtrar aristas donde proyección cae dentro del segmento
    4. Ordenar por distancia
    """
    
    # Busca KNN en estructura espacial
    _, indices = kdtree.query(p, k=20)
    
    candidatos = []
    for edge_id in get_edges_from_nodes(indices, red_viaria):
        edge = red_viaria.edges[edge_id]
        geometry = edge['geometry']
        
        # Proyectar punto a arista
        proj = project_point_to_segment(p, geometry)
        
        if proj['distance'] <= max_dist:
            candidatos.append({
                'edge_id': edge_id,
                'projection': proj,
                'distance': proj['distance'],
                'edge': edge
            })
    
    # Ordenar por distancia y retornar top-k
    return sorted(candidatos, key=lambda x: x['distance'])[:5]
```

**Complejidad**: O(log n + m) donde m es número de aristas cercanas

### Paso 2: Scoring Espacial

**Función de Distancia**:

```
S_dist(p, c) = exp(-d²/(2σ²))
```

Donde:
- d: distancia perpendicular GPS → arista
- σ: desviación estándar del ruido GPS (típicamente 5-10m)

En implementación:
```python
def spatial_score(p, candidato):
    d = candidato['distance']
    sigma = 7.0  # Desviación estándar GPS
    return math.exp(-d**2 / (2 * sigma**2))
```

### Paso 3: Scoring de Dirección

Evalúa si la geometría de la arista se alinea con la dirección del movimiento.

```
S_dir(p, p_prev, c) = cos(θ)
```

Donde θ es la diferencia angular entre:
- Vector de movimiento: v = p - p_prev
- Vector de arista: e = proj_end - proj_start

```python
def direction_score(p_current, p_prev, candidato):
    # Vector de movimiento GPS
    v_movement = normalize(p_current - p_prev)
    
    # Vector de arista
    edge_geometry = candidato['edge']['geometry']
    v_edge = normalize(edge_geometry.coords[-1] - edge_geometry.coords[0])
    
    # Coseno del ángulo
    cos_angle = dot(v_movement, v_edge)
    
    # Penalizar direcciones opuestas levemente
    return max(0, cos_angle)  # Si va al revés, score = 0
```

### Paso 4: Scoring Temporal (Viterbi)

Asegurar consistencia con punto anterior usando **algoritmo de Viterbi**:

```
P(c_i | c_{i-1}) ∝ exp(-β * d_viaria²)
```

Donde:
- d_viaria: distancia de carretera entre proyecciones
- β: factor de penalización de viajes largos

```python
def temporal_score(prev_matched, candidato, dt):
    """
    Score de compatibilidad temporal.
    Penaliza transiciones que requerirían velocidades no físicas.
    """
    prev_proj = prev_matched['projection']['point']
    curr_proj = candidato['projection']['point']
    
    # Distancia de carretera (aproximada)
    d_viaria = calculate_network_distance(prev_matched, candidato)
    
    # Velocidad requerida
    v_required = d_viaria / dt
    
    # Penalizar si v > v_max_reasonable (200 km/h ≈ 55 m/s)
    v_max = 55.0
    if v_required > v_max:
        return math.exp(-(v_required - v_max)**2 / 100)
    else:
        return 1.0
```

### Paso 5: Combinación de Scores

```python
def combine_scores(score_dist, score_dir, score_temp, 
                   w_dist=1.0, w_dir=0.5, w_temp=0.3):
    """
    Combinar múltiples scores con pesos.
    """
    return (w_dist * score_dist + 
            w_dir * score_dir + 
            w_temp * score_temp)
```

**Pesos configurables**:
- `w_dist`: 1.0 - proximidad GPS (más importante)
- `w_dir`: 0.5 - consistencia de dirección
- `w_temp`: 0.3 - consistencia temporal

---

## Formulación Matemática

### Modelo Probabilístico

El problema está formulado como un **Hidden Markov Model (HMM)**:

**Estado oculto**: Arista actual e_i ∈ E

**Observación**: Punto GPS p_i

**Probabilidad de emisión**:
```
P(p_i | e_i) = (1/Z) * exp(-dist(p_i, e_i)² / 2σ²)
```

**Probabilidad de transición**:
```
P(e_i | e_{i-1}) ∝ exp(-β * routing_distance(e_{i-1}, e_i))
```

**Objetivo**: Maximizar
```
P(e_1, e_2, ..., e_n | p_1, p_2, ..., p_n)
```

**Solución**: Algoritmo de Viterbi en O(n * |E|²)

### Matriz de Confusión (Criterios de Validación)

```
Verdadero Positivo (TP): Punto correctamente asignado
Falso Positivo (FP):     Punto asignado a carretera equivocada
Falso Negativo (FN):     Punto no asignado (filtrado)
```

**Métrica**: Precisión de match
```
Match Rate = TP / (TP + FP)
```

Típicamente: **95-99%** en carreteras bien cartografiadas

---

## Implementación

### Script Principal: `Scripts/parsers/map_matching.py`

**Funciones clave**:

```python
class GPSMapMatcher:
    
    def __init__(self, network_file=None, bbox=None):
        """Cargar o descargar red viaria."""
        
    def match_trajectory(self, gps_points, max_dist=100):
        """Procesar trayectoria completa."""
        
    def match_point(self, p, prev_match, dt):
        """Ajustar un punto individual."""
        
    def project_to_edge(self, point, edge_geometry):
        """Proyectar punto a arista."""
        
    def scoring_function(self, candidatos, point, prev_match):
        """Calcular scores combinados."""
```

### Entrada (CSV Raw)

```csv
timestamp,lat,lon,alt,gps_speed,...
2025-10-09 18:35:44.041,40.54674213,−3.617870327,614.8,21.81,...
2025-10-09 18:35:44.179,40.54673951,−3.618056401,615.1,21.92,...
```

### Salida (CSV Map-Matched)

```csv
timestamp,lat,lon,lat_raw,lon_raw,x_utm,y_utm,dist_to_road_m,road_name,highway,edge_id
2025-10-09 18:35:44.041,40.5467502,-3.6178561,40.54674213,-3.617870327,447681.66,4488626.51,2.34,Avenida Matapiñoneras,secondary,146.0
2025-10-09 18:35:44.179,40.5467480,-3.6181023,40.54673951,-3.618056401,447680.12,4488625.89,1.89,Avenida Matapiñoneras,secondary,146.0
```

**Nuevas columnas Added**:
- `lat`, `lon`: Coordenadas ajustadas a carretera
- `lat_raw`, `lon_raw`: Coordenadas GPS originales
- `x_utm`, `y_utm`: Proyección UTM ETRS89 Zona 30N
- `dist_to_road_m`: Distancia perpendicular a carretera (m)
- `road_name`: Nombre de la vía
- `road_ref`: Referencia (A-6, M-503, etc.)
- `highway`: Tipo OSM
- `edge_id`: Identificador de arista

---

## Ejemplos de Uso

### Ejemplo 1: Procesar un archivo individual

```bash
python Scripts/parsers/map_matching.py \
    --file Doback-Data/processed-data/DOBACK024_20251009.csv \
    --network output/road_network.graphml
```

### Ejemplo 2: Procesar directorio completo

```bash
python Scripts/parsers/map_matching.py \
    --input Doback-Data/processed-data \
    --output Doback-Data/map-matched \
    --glob "DOBACK024_*.csv"
```

### Ejemplo 3: Descargar red automáticamente

```bash
python Scripts/parsers/map_matching.py \
    --input Doback-Data/processed-data \
    --output Doback-Data/map-matched \
    --auto-network  # Descargar de OSM automáticamente
```

### Ejemplo 4: Ajustar parámetros

```bash
python Scripts/parsers/map_matching.py \
    --input Doback-Data/processed-data \
    --output Doback-Data/map-matched \
    --max-dist 150              # Radio de búsqueda (m)
    --dir-weight 0.7            # Peso de dirección
    --network-type drive_service  # Tipo de carreteras
```

### Siguiente etapa (features de terreno en `featured`)

```bash
python Scripts/lidar/compute_route_terrain_features.py \
    --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
    --output Doback-Data/featured/DOBACK024_20251009_seg87.csv
```

---

## Parámetros Ajustables

| Parámetro | Default | Rango | Efecto |
|-----------|---------|-------|--------|
| `max_dist` | 100 m | 50-200 | Radio de búsqueda de aristas |
| `dir_weight` | 0.5 | 0.0-1.0 | Importancia de alineación de dirección |
| `sigma_gps` | 7 m | 5-15 | Desviación estándar del ruido GPS |
| `v_max` | 55 m/s | - | Velocidad máxima razonable |
| `k_candidates` | 5 | 3-10 | Número de candidatos considerados |

---

## Casos de Uso y Limitaciones

### Funcionamiento Óptimo ✓

- ✓ Carreteras bien definidas en OpenStreetMap
- ✓ Trayectorias urbanas con espaciado temporal regular (<5s)
- ✓ Datos GPS válidos (HDOP < 2.5)
- ✓ Velocidades consistentes

### Casos Problemáticos ✗

- ✗ Carreteras nuevas no en OSM
- ✗ Zonas rurales con red vaga
- ✗ Saltos GPS grandes (>200m)
- ✗ Paradas largas
- ✗ Maniobras violentas

### Mejoras Futuras

1. **HMM continuo**: Considerar puntos intermedios sin GPS
2. **Redes multimodales**: Considerar cambios de modo (moto, coche, bici)
3. **Learning de parámetros**: Ajuste automático de pesos con ML
4. **Mapas dinámicos**: Actualización en tiempo real de red

---

## Referencias

1. **Newson, P., & Krumm, J. (2009)**. "Hidden Markov Maps: Inferring a Road Map from Raw GPS Data". Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems.

2. **Lou, Y., Zhang, C., Zheng, Y., Xie, X., Wang, W., & Huang, Y. (2009)**. "Map-Matching for Low-Sampling-Rate GPS Trajectories". GIS '09: Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems.

3. **Quddus, M. A., Ochieng, W. Y., & Noland, R. B. (2007)**. "Current map-matching algorithms for transport applications: State-of-the art and future research directions". Journal of Navigation, 60(3), 519-535.

4. **OpenStreetMap Project**: https://www.openstreetmap.org/

5. **OSMnx Documentation**: https://osmnx.readthedocs.io/


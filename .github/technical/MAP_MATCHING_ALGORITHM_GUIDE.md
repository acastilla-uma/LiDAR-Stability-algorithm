# Guía Completa: Algoritmo de Map Matching (`map_matching.py`)

## Resumen ejecutivo

El script `Scripts/parsers/map_matching.py` implementa un **map matching de tres capas** para ajustar GPS ruidoso a red viaria. Combina:
1. **Scoring geométrico**: qué tan bien encaja el GPS en cada vía candidata
2. **Memoria temporal**: bonos de continuidad + histéresis de cambio
3. **Votación post-proceso**: corrección de spikes aislados

El resultado es una trayectoria **estable**, **realista** y **altamente resistente a ruido** incluso en entornos urbanos densos.

---

## TABLA RÁPIDA: Parámetro → Problema → Solución

| Parámetro | Problema que soluciona | Efecto de SUBIR | Efecto de BAJAR |
|-----------|------------------------|-----------------|-----------------|
| `--max-dist` | edge_id=-1 (sin asignar) | ↑ Cubre más GPS lejanos | ↓ Evita vías paralelas |
| `--dir-weight` | Engancha vías perpendiculares | ↑ Demanda más alineación | ↓ Premia proximidad cruda |
| `HISTORY_WINDOW` | Parpadea rápido entre vías | ↑ Más memoria, más pegajoso | ↓ Más reactivo a giros |
| `HISTORY_DECAY` | Olvida puntos recientes | ↑ Pesos parejos, memoria larga | ↓ Drop-off fuerte, reactivo |
| `HISTORY_BONUS` | Cambia de vía fácilmente | ↑ Bonus fuerte, pegajoso | ↓ Cambios fáciles, parpadea |
| `SWITCH_THRESHOLD` | Saltos entre paralelas | ↑ Cambios difíciles (inercia) | ↓ Cambios fáciles (reactivo) |
| `SMOOTH_RADIUS` | Spikes aislados en salida | ↑ Corrige picos distantes | ↓ Preserva cambios micro |
| `SMOOTH_MIN_RUN` | Spike de 1 punto | ↑ Elimina bloques cortos | ↓ Conserva más cambios |

**Lectura:** Si ves "parpadea entre vías" → sube `HISTORY_WINDOW`, `HISTORY_BONUS`, `SWITCH_THRESHOLD`. Si "tarda en girar" → baja los mismos.

---

---

# PARTE 0: CARTA DE PROBLEMAS → SOLUCIONES

## Mapeo visual: Síntoma → Solución rápida

```
¿Qué VES?                         ¿Qué HACER?
─────────────────────────────────────────────────────────────
Saltos zigzag entre vías      → SWITCH_THRESHOLD ↑ + HISTORY_BONUS ↑
Tarda en girar en cruces      → SWITCH_THRESHOLD ↓ + HISTORY_WINDOW ↓
Muchos edge_id=-1             → --max-dist ↑ (o revisar red)
Engancha aceras/senderos      → Confirmar network_type="drive"
Zig-zag en rotondas           → HISTORY_DECAY ↑ + SMOOTH_RADIUS ↑
Ejecución muy lenta           → SAMPLE_DENSITY ↓ o --max-dist ↓
```

---

# PARTE 1: FUNDAMENTOS DEL ALGORITMO

## 1. Qué problema resuelve

### Entrada: GPS ruidoso
- Precisión: ±5 a ±50 metros (depende de número de satélites, reflejos urbanos)
- Errores sistemáticos: "saltos" a vías paralelas (aceras, carriles bici, calles contiguas)
- Discontinuidades: pérdida de satélite por segundos

### Salida esperada
- Trayectoria **snapped** a la red viaria real
- Secuencia de vías (`edge_id`) **suave** y **coherente**
- Distancias punto-a-vía (`dist_to_road_m`) **minimizadas**

### ¿Por qué es complejo?
En una esquina urbana típica hay 5–15 vías candidatas (carreteras, aceras, carriles bici, callejas). El algoritmo debe:
- Elegir la correcta **simultáneamente** para 10,000–100,000 puntos
- Mantener **consistencia** aunque haya ruido local
- Reaccionar **rápido** a giros reales sin parpadear

---

## 2. Arquitectura de tres capas

```
             GPS bruto
                ↓
        [CAPA 1: SCORING]
      Proyecta en red local
      Calcula distancia + dirección
           → candidato mejor
                ↓
     [CAPA 2: TEMPORAL ONLINE]
    Compara con historial reciente
    Aplica bonos de continuidad
    Requiere mejora mínima para cambiar
           → edge_id tentativo
                ↓
    [CAPA 3: POST-PROCESO]
       Suavizado de votación
      Elimina spikes de 1-2 puntos
      Reprojección final
           → SALIDA FINAL
```

Cada capa es **independiente y removible**. Sin capa 2 el algoritmo es rápido pero parpadea. Sin capa 3 es inestable. Las tres juntas = robusto.

---

# PARTE 2: ANÁLISIS PROFUNDO DE CADA CAPA

## Capa 1: Scoring Geométrico

### 2.1 Red viaria como grafo

La red es un grafo NetworkX donde:
- **Nodos**: intersecciones (lat/lon)
- **Aristas**: segmentos viarios con atributos:
  - `highway`: tipo OSM (primary, residential, footway, etc.)
  - `name`: nombre de vía
  - `ref`: código de ruta
  - `geometry`: shapely LineString (coordenadas UTM)

### 2.2 Indexación espacial: KD-Tree

**Necesidad:** para cada punto GPS (~100,000), encontrar vías cercanas rápidamente.

**Solución:**
```python
# Pseudocódigo
edges_sampled = []
for edge in network.edges:
    line = edge.geometry
    # Muestrear cada 0.25 metros
    for point_sampled in line.interpolate([0, 0.25, 0.5, ..., line_length]):
        edges_sampled.append({
            'point': point_sampled,
            'edge_id': edge.id,
            'highway': edge.highway
        })

kdtree = cKDTree(edges_sampled.coordinates)
```

**Complejidad:**
- Red de 6600 aristas → ~1.2M puntos indexados
- Query rango-150m: O(log n) + O(m) donde m = candidatos cerca
- Típico: m = 5–50 candidatos por GPS

### 2.3 Scoring de distancia

Para cada candidato:
1. **Proyección ortogonal** del punto GPS sobre la arista:
   ```
   Vía:        A -------- B
               
   GPS:        P
               
   P_proyectado: punto sobre AB más cercano a P
   dist = ||P - P_proyectado||
   ```

2. **Score normalizado:**
   ```
   dist_score = (dist / max_dist_m)²
   ```
   
   **¿Por qué elevar al cuadrado?**
   - Penaliza distancias grandes exponencialmente
   - GPS a 150 m: score = 1.0 (máximo)
   - GPS a 75 m (mitad): score = 0.25 (1/4, penalizado fuerte)
   - Evita que vías lejanas compitan

### 2.4 Scoring de dirección

**Premisa:** el vehículo viaja en una dirección. Si la vía va en la misma dirección, es mejor candidata.

```
heading_vía = atan2(lat_B - lat_A, lon_B - lon_A)
heading_movimiento = atan2(lat_i+1 - lat_i, lon_i+1 - lon_i)
diferencia = |heading_vía - heading_movimiento|

dir_score = (diferencia / 180°) con máximo de 1.0
```

**Interpretación:**
- Si ambos alineados (0°): dir_score = 0 (mejor)
- Si perpendiculares (90°): dir_score = 0.5
- Si opuestos (180°): dir_score = 1.0 (peor)

### 2.5 Score combinado

```
score = dist_score × (1 - dir_weight) + dir_score × dir_weight
```

**Valores típicos:**
- `dir_weight = 0.0`: puro proximidad ("pégame a la vía más cercana")
- `dir_weight = 0.35`: 65% proximidad, 35% dirección (urbano/periurbano)
- `dir_weight = 0.55`: 45% proximidad, 55% dirección (carretera abierta)
- `dir_weight = 1.0`: pura dirección ("voy en la dirección correcta, ignora distancia")

**Efecto en práctica:**
- **dir_weight bajo**: resuelve GPS que salta a paralela cercana, pero puede enganchar callejuelas
- **dir_weight alto**: resuelve tomar callesuelas, pero penaliza puntos ruidosos fuera de la vía

**Mejor candidato:** el de score **mínimo** (es costo, no probabilidad).

---

## Capa 2: Suavizado Temporal (Online)

Este es el **verdadero anti-ruido**. Funciona stream: a medida que procesa punto i, ve puntos i-1, i-2, ... i-HISTORY_WINDOW.

### 2.6 Historial con decaimiento exponencial

Se mantiene una cola (`deque`) de los últimos `HISTORY_WINDOW` edges:

```python
# Pseudocódigo
history_deque = [edge_id_1, edge_id_2, ..., edge_id_HISTORY_WINDOW]

# Pesos exponenciales
pesos = [decay^(window-1), decay^(window-2), ..., decay^0]
       = [0.70^3, 0.70^2, 0.70^1, 0.70^0]
       = [0.343, 0.49, 0.70, 1.0]
```

**Intuición:**
- Punto más reciente (i-1) tiene peso 1.0 (máxima influencia)
- Puntos antiguos (i-4) tienen peso 0.34 (influencia suave, no olvida)

### 2.7 Bonus de continuidad

Una vez el scoring geométrico elige candidato `C_best` con score `s_best`:

```python
# Ajuste de score por historial
s_adjusted = s_best

for edge_historico, peso in zip(history_deque, pesos):
    if edge_historico == C_best:
        # Rebajar artificialmente el score
        s_adjusted -= HISTORY_BONUS × peso
```

**Ejemplo numérico:**
```
Edge histórico: [10, 10, 10, 10]  (últimos 4 pts en vía 10)
Pesos:          [0.34, 0.49, 0.70, 1.0]
Candidatos scoring: vía 10 (score 0.45), vía 12 (score 0.42)

Sin bonus: gana vía 12 (0.42 < 0.45) → CAMBIO
Con bonus HISTORY_BONUS=0.45:
  vía 10: 0.45 - 0.45×(0.34+0.49+0.70+1.0) = 0.45 - 0.76 = -0.31 ← gana!
  
RESULTADO: se pega a vía 10 → continuidad
```

**Clave:** el bonus no es "rechazar cambios", es "reducir artificialmente el score de continuación". Si cambio real es fuerte (ej: giro a 90°), el scoring geométrico lo detecta igual.

### 2.8 Histéresis de cambio (SWITCH_THRESHOLD)

Incluso si `C_best` tiene score mejor, no cambies a menos que la **mejora relativa** sea suficiente:

```python
current_edge = history_deque[-1]
score_mejora = (score_current - score_best) / score_current

if score_mejora < SWITCH_THRESHOLD:
    return current_edge  # Rechazar cambio
else:
    return C_best        # Aceptar cambio
```

**Ejemplo:**
```
Edge actual (vía 10): score 0.50
Candidato (vía 12): score 0.48 (0.02 pts mejor)
Mejora relativa = 0.02 / 0.50 = 4%
SWITCH_THRESHOLD = 0.4 (40%)

4% < 40% → RECHAZAR CAMBIO → mantener vía 10

Necesitaría: score_mejora ≥ 0.40 × 0.50 = 0.20 pts mejora
→ candidato debería ser 0.30 o mejor
```

**Efecto:**
- Alto (0.4+): conservador, mantiene vía actual incluso si vecina es marginalmente mejor
- Bajo (0.1-): reactivo, cambia por mejoras pequeñas
- Típico: 0.35–0.45 en urbano

### 2.9 Interacción HISTORY_BONUS vs SWITCH_THRESHOLD

Estos son **dos frentes** diferentes de "anti-saltos":

| Parámetro | Mecanismo | Cuándo se aplica |
|-----------|-----------|-----------------|
| HISTORY_BONUS | Rebaja score de edge continuación | Siempre en candidato |
| SWITCH_THRESHOLD | Rechaza cambios <40% mejora | Al decidir cambio |

**Analogía:** HISTORY_BONUS es "hacer la meta más ancha". SWITCH_THRESHOLD es "pedir más puntos para ganar".

---

## Capa 3: Suavizado Post-Proceso (Votación Vecinal)

Funciona **después** de procesar toda la secuencia. Es correctivo, no preventivo.

### 2.10 Votación por ventana deslizante

```python
for i in range(len(edge_sequence)):
    # Ventana [i-SMOOTH_RADIUS : i+SMOOTH_RADIUS]
    vecinos = edge_sequence[i - SMOOTH_RADIUS : i + SMOOTH_RADIUS + 1]
    
    # Qué edge domina?
    edge_dominante, cuenta = Counter(vecinos).most_common(1)[0]
    
    # Si no es el actual, y el bloque actual es corto, rectificar
    if edge_dominante != edge_sequence[i]:
        if longitud_bloque_actual <= SMOOTH_MIN_RUN:
            edge_sequence[i] = edge_dominante
```

### 2.11 Ejemplo visual

```
Secuencia input:  [10, 10, 10, 15, 10, 10, 10, 12, 12, 12]
                  (la vía 15 está aislada a posición 3)

Ventana SMOOTH_RADIUS=2:
  i=3 (edge=15): vecinos [10, 10, 15, 10, 10] → dominante=10, bloque_15=1 punto
  SMOOTH_MIN_RUN=1 → se cumple → CAMBIAR a 10

Secuencia output: [10, 10, 10, 10, 10, 10, 10, 12, 12, 12]
                   ✓ Spike eliminado
```

### 2.12 Convergencia iterativa

Se repite el proceso hasta máximo 4 pasadas:

```
Pasada 1: detecta spikes de 1 punto
Pasada 2: detecta spikes que quedan después de P1
Pasada 3: pequeños ajustes finales
Pasada 4: asegurar convergencia
```

Típicamente converge en 1–2 pasadas. Las 4 son seguridad.

---

# PARTE 3B: MATRIZ RÁPIDA - PARÁMETRO → PROBLEMA → SOLUCIÓN

## Entendimiento directo de cada parámetro

### 1. `--max-dist` (radio de búsqueda)

**Problema que soluciona:** "Tengo muchos puntos sin asignar (edge_id=-1)"

**Cómo funciona:** Define el radio máximo alrededor de cada GPS donde buscar vías candidatas.

**Efecto:**
- ⬆️ Subirlo (150 → 200): GPS muy lejano tiene oportunidad, cubre zonas con red dispersa
- ⬇️ Bajarlo (150 → 100): Solo vías cercanas se consideran, evita vías paralelas lejanas

**Cuándo usar:**
- **Subir**: muchos edge_id=-1 en zonas con vías dispersas o GPS muy ruidoso
- **Bajar**: saltos constantes a vías paralelas/aceras que están lejos

---

### 2. `--dir-weight` (peso de dirección vs distancia)

**Problema que soluciona:** "Se está metiendo en callejas perpendiculares / engancha desviaciones"

**Cómo funciona:** Mezcla scoring de distancia + dirección.
```
score = dist × (1 - dir_weight) + dirección × dir_weight
```

**Efecto:**
- ⬆️ Subirlo (0.35 → 0.50): demanda más alineación → evita perpendiculares, giros raros
- ⬇️ Bajarlo (0.35 → 0.20): premia proximidad cruda → pega a vía más cercana sin importar ángulo

**Cuándo usar:**
- **Subir**: urbano denso con muchos cruces, no quieres desviaciones
- **Bajar**: carretera recta, GPS ruidoso con ángulos erráticos

---

### 3. `HISTORY_WINDOW` (cuántos puntos atrás mirar)

**Problema que soluciona:** "Parpadea entre dos vías (zigzag) cada 1-2 puntos"

**Cómo funciona:** Guarda últimos N edge_ids y los usa para "votación de continuidad".

**Efecto:**
- ⬆️ Subirlo (4 → 6): más puntos atrás influyen → más "pegajoso", menos cambios
- ⬇️ Bajarlo (4 → 2): menos memoria → más sensible a cambios, más reactivo

**Cuándo usar:**
- **Subir**: saltos repetitivos entre dos vías paralelas
- **Bajar**: tarda mucho en reconocer giros reales

---

### 4. `HISTORY_DECAY` (cómo decaen puntos antiguos)

**Problema que soluciona:** "La continuidad no se mantiene bien / cae bruscamente"

**Cómo funciona:** Exponencial `peso = decay^(distancia_atrás)`. 0.70 = puntos viejos pierden 30% influencia.

**Efecto:**
- ⬆️ Subirlo (0.70 → 0.85): puntos viejos mantienen peso → memoria "pareja"
- ⬇️ Bajarlo (0.70 → 0.50): puntos viejos pierden rápido → "drop-off" fuerte

**Cuándo usar:**
- **Subir**: quieres transiciones suaves, no caídas bruscas de influencia
- **Bajar**: quieres que recientes dominen totalmente, antiguos sin influencia

---

### 5. `HISTORY_BONUS` (fuerza del bonus de "pegajosidad")

**Problema que soluciona:** "Cambia de vía muy fácil por ruido local"

**Cómo funciona:** Si siguiente candidato = edge anterior, rebaja score artificialmente (lo favorece).

**Efecto:**
- ⬆️ Subirlo (0.45 → 0.60): bonus fuerte → ultra-pegajoso, difícil cambiar
- ⬇️ Bajarlo (0.45 → 0.25): bonus débil → cambios fáciles, reactivo

**Cuándo usar:**
- **Subir**: muchos saltos causados por ruido local, quieres que se quede
- **Bajar**: algoritmo "atrapado" en vía anterior, necesita reaccionar a giro real

---

### 6. `SWITCH_THRESHOLD` (mejora mínima relativa para cambiar)

**Problema que soluciona:** "Cambia de vía por mejoras marginales" o "no cambia para giros reales"

**Cómo funciona:** Exige mejora RELATIVA mínima: `(score_actual - score_mejor) / score_actual`

**Efecto:**
- ⬆️ Subirlo (0.40 → 0.55): cambios MUY difíciles (necesita 55% mejora), ultra-conservador
- ⬇️ Bajarlo (0.40 → 0.20): cambios fáciles (basta 20% mejora), sensible a ruido

**Cuándo usar:**
- **Subir**: "saltos entre dos vías cada segundo", quieres máxima inercia
- **Bajar**: "no reconoce giros", tarda mucho tiempo en cambiar

---

### 7. `SMOOTH_RADIUS` (radio de votación post-proceso)

**Problema que soluciona:** "Hay spikes de 1-2 puntos mal asignados en la salida final"

**Cómo funciona:** Mira ventana `[i-2, i-1, i, i+1, i+2]`, si mayoría votó por otro edge, cambia.

**Efecto:**
- ⬆️ Subirlo (2 → 3-4): ventana grande, corrige picos aislados bien
- ⬇️ Bajarlo (2 → 1): ventana pequeña, casi no corrige, preserva cambios micro

**Cuándo usar:**
- **Subir**: ves spikes aislados de 1-2 puntos rodeados de otro edge
- **Bajar**: cambios reales rápidos (raro si GPS es denso)

---

### 8. `SMOOTH_MIN_RUN` (longitud mínima de bloque para conservar)

**Problema que soluciona:** "Post-proceso está eliminando cambios legítimos de corta duración"

**Cómo funciona:** Si bloque tiene ≤ N puntos Y vecinos votaron por otro edge → cambiar.

**Efecto:**
- ⬆️ Subirlo (1 → 2-3): bloques de 1 punto se eliminan, 2+ se conservan (agresivo)
- ⬇️ Bajarlo (1 → 0): nada se elimina (desactiva post-proceso)

**Cuándo usar:**
- **Subir**: mucho ruido de 1 punto, quieres post-proceso agresivo
- **Bajar**: cambios reales rápidos que duran 1 punto (rara vez)

---

# PARTE 3: PARÁMETROS DETALLADOS

## 3. Parámetros configurables

### Categoría A: Búsqueda geométrica

#### `DEFAULT_MAX_DIST_M` (CLI: `--max-dist`, defecto: 150)

**Qué es:** radio máximo de búsqueda de candidatos

**Fórmula de impacto:**
- Distancia < max_dist_m: candidato considerado
- Distancia > max_dist_m: candidato rechazado

**Efecto de subirlo (ej: 150 → 250):**
- ✓ Roba puntos lejanos legítimos (GPS malo)
- ✗ Aumenta riesgo de engancharse en vía paralela
- ✗ Más computación (más candidatos)

**Efecto de bajarlo (ej: 150 → 80):**
- ✓ Evita vías paralelas lejanas
- ✗ Riego de edge_id=-1 (sin asignación)
- ✓ Computación rápida

**Rango orientativo por contexto:**

| Contexto | Rango (m) |
|----------|----------|
| Urbano denso (avenidas 5 pts) | 50–80 |
| Urbano mixto (calles+avenidas) | 80–130 |
| Periurbano (menos vías) | 120–180 |
| GPS muy ruidoso (sierra sin satélites) | 180–250 |
| Carretera abierta | 100–150 |

**Diagnóstico:**
- Muchos edge_id=-1 → subir max_dist
- Muchos saltos a vías paralelas → bajar max_dist

#### `DEFAULT_DIR_WEIGHT` (CLI: `--dir-weight`, defecto: 0.35)

**Qué es:** peso de dirección vs distancia en scoring

**Fórmula:**
```
score = dist_score × (1 - dir_weight) + dir_score × dir_weight
```

**Valores extremos:**
- `0.0`: score = dist_score (puro proximidad)
- `1.0`: score = dir_score (pura dirección)
- `0.35`: 65% distancia, 35% dirección

**Efecto de subirlo (0.35 → 0.50):**
- Demanda más alineación direccional
- ✓ Evita vías que van perpendiculares
- ✗ Puede fallar en curvas si GPS sale tangente

**Efecto de bajarlo (0.35 → 0.20):**
- Premia proximidad cruda
- ✓ Más robusto a falta de satélites
- ✗ Puede engancharse en aceras

**Ajuste recomendado por caso:**

| Caso | Rango |
|------|-------|
| Urbano denso, muchos giros | 0.35–0.50 |
| Carretera recta | 0.25–0.35 |
| GPS indeterminista (sendero) | 0.15–0.25 |
| Rotondas complejas | 0.40–0.55 |

#### `SAMPLE_DENSITY` (defecto: 0.25 metros)

**Qué es:** espaciado de puntos de muestreo en aristas

**Complejidad:**
- Valor 0.25 m: ~4 puntos/metro → muy preciso, ~1.2M puntos KD-tree
- Valor 0.5 m: ~2 puntos/metro → preciso, ~600K puntos
- Valor 1.0 m: ~1 punto/metro → rápido, ~300K puntos

**Efecto de subirlo (0.25 → 0.5):**
- ✓ Menos RAM, búsquedas KD-tree más rápidas
- ✗ Proyecciones menos precisas en aristas largas
- Típicamente impacto pequeño en resultado

**Efecto de bajarlo (0.25 → 0.1):**
- ✓ Proyecciones ultraprecisas
- ✗ RAM explota, muy lento
- Rara vez vale la pena

**Recomendación:** dejar en 0.25 salvo que procesamiento sea extremadamente lento

---

### Categoría B: Red viaria

#### `EXCLUDED_HIGHWAY_TYPES` (defecto: 15 tipos)

```python
EXCLUDED_HIGHWAY_TYPES = {
    'footway',        # Aceras
    'path',           # Senderos peatonales
    'cycleway',       # Carriles bici
    'pedestrian',     # Plazas peatonales
    'steps',          # Escaleras
    'track',          # Caminos sin pavimentar
    'bridleway',      # Caminos de caballos
    'corridor',       # Pasillos (ej: polígonos)
    'via_ferrata',    # Vías de escalada
    'proposed',       # Vías propuestas (OSM)
    'construction',   # En construcción
    'raceway',        # Circuitos de carreras
    'rest_area',      # Áreas de descanso (no movimiento)
    'elevator',       # Ascensores (OSM meta)
    'bus_guideway'    # Carriles de autobús (no sé si aplicable)
}
```

**¿Cuándo añadir?**
- Si el algoritmo pone puntos en callejas privadas: añade `service` o `private`
- Si pone en escaleras/rampas: la lista ya cubre

**¿Cuándo remover?**
- Si está excluyendo demasiado (muchos edge_id=-1)
- Típicamente no: la lista es conservadora

**Impacto de filtrar:**
```
Red antes: 27,905 aristas
Red después: 6,646 aristas (76% reducción)

Tiempo procesamiento:
  Antes: 42 segundos (100,000 GPS)
  Después: 16 segundos (62% más rápido)

Saltos entre vías paralelas:
  Antes: 12 saltos en ruta de 2 km
  Después: 1 salto legítimo
```

#### `network_type="drive"` (en osmnx)

Además de filtrar tipos, OSMnx descarga sólo vías "drivable":
- Ignora completamente footway, cycleway, etc desde OSM
- Reduce descarga 40%

---

### Categoría C: Temporales (el corazón de la estabilidad)

#### `HISTORY_WINDOW` (defecto: 4 puntos)

**Qué es:** cuántos puntos atrás se mira para "historial"

Interpretación temporal:
- Si GPS a 1 Hz: 4 puntos = 4 segundos de memoria
- Si GPS a 10 Hz: 4 puntos = 0.4 segundos de memoria

**Efecto de subirlo (4 → 6):**
- ✓ Memoria más larga, más "pegajoso"
- ✓ Evita saltos por ruido de corta duración
- ✗ Tarda más en reconocer giro real
- ✗ Más "inercia" en cambios

**Efecto de bajarlo (4 → 2):**
- ✓ Reactivo a giros
- ✗ Sensible a ruido local

**Recomendado por tipo de ruta:**

| Tipo | HISTORY_WINDOW |
|------|--------|
| Baja velocidad urbana (10 Hz) | 4–6 |
| Velocidad media (5 Hz) | 4–5 |
| Carretera rápida (1 Hz) | 3–4 |

#### `HISTORY_DECAY` (defecto: 0.70)

**Qué es:** factor de decaimiento exponencial

Función de pesos:
```
peso[i] = decay ^ (window_size - i)

decay=0.70: [0.70^3, 0.70^2, 0.70^1, 0.70^0] = [0.34, 0.49, 0.70, 1.0]
decay=0.50: [0.50^3, 0.50^2, 0.50^1, 0.50^0] = [0.12, 0.25, 0.50, 1.0]
decay=0.90: [0.90^3, 0.90^2, 0.90^1, 0.90^0] = [0.73, 0.81, 0.90, 1.0]
```

**Efecto de subirlo (0.70 → 0.85):**
- Puntos antiguos tienen más influencia
- ✓ Memoria "más pareja", menos drop-off
- ✗ Menos reacción a cambios recientes

**Efecto de bajarlo (0.70 → 0.50):**
- Puntos antiguos tienen muy poca influencia
- ✓ Memoria corta, más reactiva
- ✗ Más sensible a ruido

**Rangos típicos:** 0.50–0.90

#### `HISTORY_BONUS` (defecto: 0.45)

**Qué es:** fuerza del bonus de continuidad

Impacto directo:
```
ajuste = -HISTORY_BONUS × peso_historico
```

**Efecto de subirlo (0.45 → 0.60):**
- Bonus más fuerte
- ✓ Pega más a vía anterior
- ✗ Dificulta giros reales

**Efecto de bajarlo (0.45 → 0.25):**
- Bonus débil, casi ignorado
- ✓ Cambios fáciles
- ✗ Parpadea en ruido

**Rango:** 0.25–0.65

#### `SWITCH_THRESHOLD` (defecto: 0.40)

**Qué es:** mejora mínima **relativa** para cambiar de vía

Decisión:
```
if (score_actual - score_candidato) / score_actual < SWITCH_THRESHOLD:
    rechazar cambio
else:
    aceptar cambio
```

**Efecto de subirlo (0.40 → 0.60):**
- Cambios difíciles (necesitan 60% mejora)
- ✓ Ultraconsercador, casi inmóvil
- ✗ Tarda en giros

**Efecto de bajarlo (0.40 → 0.20):**
- Cambios fáciles (basta 20% mejora)
- ✓ Reactivo a giros
- ✗ Parpadea por ruido

**Problemas comunes:**

| Síntoma | Valor actual | Ajuste |
|---------|-------------|--------|
| Parpadea entre vías | 0.20 | → 0.35–0.40 |
| Tarda en girar | 0.50 | → 0.30–0.35 |
| Saltos aislados | 0.30 | → 0.40–0.45 |

---

### Categoría D: Post-proceso

#### `SMOOTH_RADIUS` (defecto: 2 puntos)

**Qué es:** radio de votación en post-proceso

Efecto:
```
Ventana de votación = [i-2, i-1, i, i+1, i+2]
```

**Efecto de subirlo (2 → 3):**
- Ventana más grande
- ✓ Corrige más spikes
- ✗ Puede perder cambios legítimos rápidos

**Efecto de bajarlo (2 → 1):**
- Ventana pequeña
- ✓ Preserva cambios micro
- ✗ Menos corrección

#### `SMOOTH_MIN_RUN` (defecto: 1 punto)

**Qué es:** longitud mínima de bloque para conservarlo

Decisión:
```
Si bloque_actual <= SMOOTH_MIN_RUN:
    cambiar a dominante
else:
    conservar
```

**Efecto de subirlo (1 → 2):**
- Bloquetes de 1 punto se eliminan, 2+ se conservan
- ✓ Más agresivo contra spikes
- ✗ Puede perder giros reales de 1 punto (raro si GPS es denso)

**Efecto de bajarlo (1 → 0):**
- Nada se elimina
- Prácticamente desactiva post-proceso

---

# PARTE 4: RECETARIO DE AJUSTES POR PROBLEMÁTICA

## Caso 1: Saltos entre vías paralelas (zig-zag)

**Síntoma:** GPS va recto pero aparece "diente de sierra" saltando entre calle y acera/paralela.

```
Ejemplo: Avenida principal + acera derecha, a 15 m de distancia

Pts:  1     2     3     4     5     6
Edge: [A,   A,   B,    A,   A,   B]   ← Parpadea entre A y B

Trayectoria visual: zigzag
```

### Solución ordenada:

**Paso 1: Aumentar SWITCH_THRESHOLD**
```
Default: 0.40  → Prueba: 0.50
```
- Si siguen los saltos, continúa

**Paso 2: Aumentar HISTORY_WINDOW**
```
Default: 4     → Prueba: 5 ó 6
```
- Más memoria = más "inercia"

**Paso 3: Aumentar HISTORY_BONUS**
```
Default: 0.45  → Prueba: 0.55
```
- Bonus más fuerte = pega más a vía anterior

**Paso 4: Aumentar SMOOTH_MIN_RUN**
```
Default: 1     → Prueba: 2
```
- Post-proceso más agresivo

**Paso 5: Bajar --max-dist**
```
Default: 150   → Prueba: 100
```
- Si la paralela está muy lejos, excluirla de candidatos

### Ejemplo real (Doback024, 20250929):
```
Síntoma: spikes en rotonda, 12 cambios por 500 metros

Ajustes aplicados:
  SWITCH_THRESHOLD: 0.40 → 0.45
  HISTORY_BONUS: 0.45 → 0.50
  SMOOTH_RADIUS: 2 → 3

Resultado: 2 cambios en 500 metros (1 legítimo entrada, 1 salida)
```

---

## Caso 2: Tarda en reconocer giro real

**Síntoma:** en cruces/rotondas, se queda pegado a la vía anterior

```
Ruta real:    [Calle A] → [giro] → [Calle B]

Asignado:     [...A, A, A, A, A, B, B, B...]
                          ^ tarda aquí
```

### Solución ordenada:

**Paso 1: Bajar SWITCH_THRESHOLD**
```
Default: 0.40  → Prueba: 0.30
```
- Cambios más fáciles

**Paso 2: Bajar HISTORY_WINDOW**
```
Default: 4     → Prueba: 3
```
- Menos "memoria", más reactivo

**Paso 3: Bajar HISTORY_BONUS**
```
Default: 0.45  → Prueba: 0.30
```
- Bonus débil, permite cambios

**Paso 4: Bajar SMOOTH_RADIUS (ligeramente)**
```
Default: 2     → Prueba: 1
```
- Post-proceso menos suavizador

### Validación:
Comparar número de frames hasta reconocer giro real en misma ruta.

---

## Caso 3: Muchos puntos sin asignar (edge_id=-1)

**Síntoma:** huecos en la trayectoria, distancias muy altas.

```
Pts:     1    2    3    4    5    6
Edge: [10,  -1,  -1,  12,  12,  -1]  ← Muchos blancos
```

### Causas posibles:

1. **Red demasiado pequeña** → bbox no cubre
   - Solución: Subir `NETWORK_BUFFER_DEG` o bajar `max_dist`

2. **max_dist demasiado pequeño** → candidatos no alcanzan
   - Solución: `--max-dist 100 → 150+`

3. **Filtrado demasiado agresivo** → tipos OSM correctos están excluidos
   - Diagnóstico: revisar qué tipo es la vía real en OSM
   - Solución: remover de `EXCLUDED_HIGHWAY_TYPES` si es imprescindible

### Solución ordenada:

**Paso 1: Subir --max-dist**
```bash
python ... --max-dist 200
```

**Paso 2: Revisar EXCLUDED_HIGHWAY_TYPES**
```python
# Si muchos -1 en periferias:
EXCLUDED_HIGHWAY_TYPES.discard('track')  # Senderos pueden ser válidos
EXCLUDED_HIGHWAY_TYPES.discard('terracería')  # Si es tipo local
```

**Paso 3: Bajar SAMPLE_DENSITY (último recurso)**
```python
SAMPLE_DENSITY = 0.5  # Menos puntos, pero query sigue coincidiendo
```

---

## Caso 4: Se mete en caminos no vehiculares

**Síntoma:** puntos asignados a senderos, aceras, carriles bici

```
Pts reales:      [Calle A, Calle A, Calle A]
Asignado:        [Acera, Calle A, Acera]  ← Saltos a no-vehícular
```

### Causa: OSM tiene tipos especiales locales

### Solución:

**Paso 1: Verificar que network_type="drive"**
- Debe estar en `load_network_from_osmnx()`

**Paso 2: Buscar qué tipo OSM es la vía problemática**
```bash
# Inspeccionar OSM manualmente o:
# En map_matching.py, añadir print de highway type
```

**Paso 3: Añadir a EXCLUDED_HIGHWAY_TYPES**
```python
EXCLUDED_HIGHWAY_TYPES.add('tipo_problematico')
```

**Paso 4: Regenerar caché**
```bash
rm -r output/cached_networks/
python Scripts/parsers/map_matching.py --glob "..." --no-cache
```

---

## Caso 5: Zig-zag en rotondas

**Síntoma:** dentro de glorieta, microcambios rápidos entre las diferentes aristas de circulación

```
Rotonda de 4 salidas:
Edge: [10, 11, 10, 12, 11, 12, 10, ...]
           ↑ Parpadea dentro la glorieta
```

### Causa: GPS ruidoso en zona circular

### Solución:

**Paso 1: Aumentar HISTORY_DECAY**
```
Default: 0.70  → Prueba: 0.80
```
- Pesos más parejos, menos drop-off

**Paso 2: Aumentar HISTORY_WINDOW**
```
Default: 4     → Prueba: 6
```
- Más memoria dentro de glorieta

**Paso 3: Aumentar SMOOTH_RADIUS**
```
Default: 2     → Prueba: 3 ó 4
```
- Post-proceso más agresivo

**Paso 4 (si sigue):** Bajar --dir-weight ligeramente
```
Default: 0.35  → Prueba: 0.25
```
- En rotonda las direcciones cambian, no penalizar tanto

---

## Caso 6: Ejecución extremadamente lenta

**Síntoma:** procesamiento tarda minutos para pocos miles de puntos

### Causa posible: red muy grande o incompatible

### Solución ordenada:

**Paso 1: Bajar SAMPLE_DENSITY**
```python
SAMPLE_DENSITY = 0.5  # De 0.25 a 0.5
```
- KD-tree 50% más pequeño

**Paso 2: Bajar --max-dist**
```bash
--max-dist 100  # De 150 a 100
```
- Menos candidatos por punto

**Paso 3: Reducir --network-buffer**
```python
NETWORK_BUFFER_DEG = 0.01  # De 0.02 a 0.01
```
- Red más pequeña

**Paso 4: Procesar por lotes**
```bash
python Scripts/parsers/map_matching.py --glob "DOBACK024_20250929_000_*.csv"  # Solo primeros 10
```
- Paralelizable

---

# PARTE 5: PROCEDIMIENTO DE CALIBRACIÓN

### Recomendado para nueva zona/vehículo:

1. **Seleccionar rutas de test (3):**
   - 1 urbana densa (muchos giros)
   - 1 carretera abierta
   - 1 mixta (periurbana)

2. **Establecer métricas base:**
   ```
   % puntos con edge_id ≠ -1  (debe ser > 99%)
   Media + p95 dist_to_road_m  (debe ser < 5 m típico)
   Nº cambios edge_id por km  (debe ser razonable para ruta)
   ```

3. **Tuning fase 1: geométrico (fijo post-process)**
   - Variar `--max-dist` en rango 80–180
   - Variar `--dir-weight` en rango 0.25–0.50
   - Elegir combo que minimize edge_id=-1 y dist_media

4. **Tuning fase 2: temporal (fijo post-process)**
   - Variar `HISTORY_*` y `SWITCH_THRESHOLD`
   - Objetivo: reducir parpadeo sin impactar giros

5. **Tuning fase 3: post-proceso (últimos ajustes)**
   - Variar `SMOOTH_RADIUS` y `SMOOTH_MIN_RUN`
   - Validar no se pierden giros legítimos

6. **Validación final:**
   - Procesar todas las rutas
   - Visualizar en mapa (ver si trayectoria es coherente)
   - Comparar metrics con baseline

---

# PARTE 6: CONFIGURACIÓN ACTUAL Y COMANDOS

## Parámetros actuales en el script:

```python
DEFAULT_MAX_DIST_M = 150.0          # Urbano mixto
DEFAULT_DIR_WEIGHT = 0.35           # 65% distancia, 35% dirección
SAMPLE_DENSITY = 0.25               # Densidad KD-tree
NETWORK_BUFFER_DEG = 0.02           # Buffer para OSM download
HISTORY_WINDOW = 4                  # 4 puntos atrás
HISTORY_DECAY = 0.70                # Decaimiento exponencial
HISTORY_BONUS = 0.45                # Bonus continuidad
SWITCH_THRESHOLD = 0.4              # Histéresis 40%
SMOOTH_RADIUS = 2                   # Votación vecinal ±2
SMOOTH_MIN_RUN = 1                  # Elimina spikes 1 punto
```

Estos valores son **buen punto de partida** para urbano/periurbano madrid. Adaptables según caso.

## Comandos útiles:

### Procesar un día completo (conservador contra saltos):
```bash
python Scripts/parsers/map_matching.py \
  --glob "Doback-Data/processed-data/DOBACK024_20250929_*.csv" \
  --max-dist 120 \
  --dir-weight 0.40
```

### Mismo día, tolerante a GPS ruidoso:
```bash
python Scripts/parsers/map_matching.py \
  --glob "Doback-Data/processed-data/DOBACK024_20250929_*.csv" \
  --max-dist 180 \
  --dir-weight 0.30
```

### Forzar recomputación (ignor caché):
```bash
python Scripts/parsers/map_matching.py \
  --glob "Doback-Data/processed-data/DOBACK024_20250929_*.csv" \
  --no-cache
```

### Usar red local específica:
```bash
python Scripts/parsers/map_matching.py \
  --glob "..." \
  --network LiDAR-Maps/geo-mad/road_network.graphml
```

---

# PARTE 7: TROUBLESHOOTING AVANZADO

## "edge_id=-1 en bordes de dataset"

**Causa:** bbox calculado no cubre puntos finales

**Fix:**
1. Subir `NETWORK_BUFFER_DEG` → `0.05` (por defecto 0.02)
2. O usar `--network` con GraphML local que ya cubre

## "Lento incluso con SAMPLE_DENSITY=0.5"

**Check:**
```bash
# Contar puntos en KD-tree
python -c "print(kdtree.data.shape[0])"  # Debe ser < 800K
```

**Fix:** Si > 1M:
- Usar `NETWORK_BUFFER_DEG = 0.01` (red más pequeña)
- Procesar por zonas con `--network` local

## "Saltos alternados entre dos vías (no desaparece con ajustes)"

**Probablemente:**
- Las dos vías tienen scores idénticos por geometría
- Post-proceso no es suficiente

**Fix:**
- Revisarocument de OSM si capturan el mismo camino
- Si capturan distintos: es ambigüedad real, aceptar

---

# CONCLUSIÓN

El algoritmo de map matching es **robusto pero configurable**. La configuración por defecto funciona bien en la mayoría de escenarios urbanos/periurbanos. Para casos especiales (GPS muy ruidoso, zonas complejas), los parámetros ofrecen control fino de tres aspectos ortogonales:

1. **Geometría:** qué tan lejos buscar, cuánto pedir dirección
2. **Temporalidad:** cuánta memoria, qué tan pegajoso, cuán fácil cambiar
3. **Suavizado:** qué tan agresivo contra spikes aislados

Aprender a ajustarse según síntomas es el camino a trayectorias estables Y precisas.

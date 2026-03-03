# Guía MAP Matching - Versión Compacta

## ¿Qué hace?
Convierte GPS ruidoso en trayectoria suave ajustada a red viaria.

**Entrada:** GPS lat/lon + ruido  
**Salida:** puntos snapped en carreteras + distancia a vía + tipo vía

---

## Cómo funciona (3 capas)

```
GPS → [Capa 1: Scoring geométrico]
    → [Capa 2: Memoria temporal + histéresis]
    → [Capa 3: Votación post-proceso]
    → SALIDA FINAL
```

### Capa 1: Scoring
- Busca vías dentro de radio `max_dist`
- Proyecta GPS ortogonalmente en cada vía
- Calcula: `score = distancia×(1-dir_weight) + dirección×dir_weight`
- Elige vía con score MÍNIMO

### Capa 2: Temporal
- Historial de últimos N puntos con pesos exponenciales
- Bonus de continuidad para mantener vía actual
- Histéresis: rechaza cambios menores

### Capa 3: Post-proceso
- Elimina spikes de 1-2 puntos aislados
- Votación vecinal

---

## TABLA RÁPIDA: 8 Parámetros

| Parámetro | Defecto | ⬆️ Sube | ⬇️ Baja |
|-----------|---------|---------|---------|
| `--max-dist` | 150 m | Cubre GPS lejano | Evita paralelas |
| `--dir-weight` | 0.35 | Demanda alineación | Pura proximidad |
| `HISTORY_WINDOW` | 4 | Más pegajoso | Más reactivo |
| `HISTORY_DECAY` | 0.70 | Peso parejo | Drop-off fuerte |
| `HISTORY_BONUS` | 0.45 | Ultrapegajoso | Cambios fáciles |
| `SWITCH_THRESHOLD` | 0.40 | Cambios difíciles | Cambios fáciles |
| `SMOOTH_RADIUS` | 2 | Corrige picos | Preserva cambios |
| `SMOOTH_MIN_RUN` | 1 | Elimina spikes | Conserva más |

---

## SÍNTOMA → SOLUCIÓN RÁPIDA

| Síntoma | Solución |
|---------|----------|
| **Zigzag (parpadea entre vías)** | ⬆️ SWITCH_THRESHOLD, HISTORY_BONUS, HISTORY_WINDOW |
| **Tarda en girar** | ⬇️ SWITCH_THRESHOLD, HISTORY_WINDOW, HISTORY_BONUS |
| **Muchos edge_id=-1** | ⬆️ --max-dist (150→200) |
| **Engancha aceras/senderos** | Verificar network_type="drive" |
| **Zigzag en rotondas** | ⬆️ HISTORY_DECAY, HISTORY_WINDOW, SMOOTH_RADIUS |
| **Muy lento** | ⬇️ SAMPLE_DENSITY o ⬇️ --max-dist |

---

## PARÁMETROS EXPLICADOS (ultra-corto)

### `--max-dist` (150)
**¿Qué?** Radio búsqueda de candidatos  
**⚠️ Problema:** muchos edge_id=-1  
**✓ Solución:** subir 150→200  
**⚠️ Riesgo:** engancharse en paralelas

### `--dir-weight` (0.35)
**¿Qué?** Peso de dirección vs distancia (65/35)  
**⚠️ Problema:** engancha desviaciones perpendiculares  
**✓ Solución:** subir 0.35→0.50  
**⚠️ Riesgo:** falla si GPS ángulo errado

### `HISTORY_WINDOW` (4)
**¿Qué?** Cuántos puntos atrás recordar  
**⚠️ Problema:** parpadea entre dos vías  
**✓ Solución:** subir 4→6  
**⚠️ Riesgo:** tarda en giros

### `HISTORY_DECAY` (0.70)
**¿Qué?** Decaimiento exponencial de influencia  
**⚠️ Problema:** influencia cae muy rápido  
**✓ Solución:** subir 0.70→0.85  
**⚠️ Riesgo:** muy reactivo si baja

### `HISTORY_BONUS` (0.45)
**¿Qué?** Rebaja score si edge = anterior  
**⚠️ Problema:** cambia de vía por ruido  
**✓ Solución:** subir 0.45→0.60  
**⚠️ Riesgo:** parpadea si baja

### `SWITCH_THRESHOLD` (0.40)
**¿Qué?** Mejora mínima relativa para cambiar  
**⚠️ Problema:** zigzag OR tarda en girar  
**✓ Solución:** subir si zigzag (→0.50), bajar si lento (→0.25)  
**⚠️ Riesgo:** muy sensible

### `SMOOTH_RADIUS` (2)
**¿Qué?** Radio votación post-proceso  
**⚠️ Problema:** spikes 1-2 puntos  
**✓ Solución:** subir 2→3  
**⚠️ Riesgo:** elimina cambios reales rápidos

### `SMOOTH_MIN_RUN` (1)
**¿Qué?** Longitud mínima bloque para conservar  
**⚠️ Problema:** elimina cambios reales de 1 punto  
**✓ Solución:** bajar 1→0  
**⚠️ Riesgo:** deja spikes

---

## PROCEDIMIENTO RÁPIDO

1. Detecta síntoma ↔ busca en tabla "SÍNTOMA → SOLUCIÓN"
2. Lee parámetro → entiende qué es
3. Ajusta valor → sube/baja según indica
4. Procesa 1-2 rutas ejemplo
5. Valida: mira estabilidad en mapa

---

## COMANDOS COMUNES

**Conservador (evita saltos):**
```bash
python Scripts/parsers/map_matching.py --glob "DOBACK024_*.csv" --max-dist 120 --dir-weight 0.40
```

**Tolerante GPS ruidoso:**
```bash
python Scripts/parsers/map_matching.py --glob "DOBACK024_*.csv" --max-dist 180 --dir-weight 0.30
```

**Sin caché:**
```bash
python Scripts/parsers/map_matching.py --glob "..." --no-cache
```

---

## CONFIG ACTUAL (DEFAULT)

```python
DEFAULT_MAX_DIST_M = 150.0          
DEFAULT_DIR_WEIGHT = 0.35           
SAMPLE_DENSITY = 0.25               
NETWORK_BUFFER_DEG = 0.02           
HISTORY_WINDOW = 4                  
HISTORY_DECAY = 0.70                
HISTORY_BONUS = 0.45                
SWITCH_THRESHOLD = 0.4              
SMOOTH_RADIUS = 2                   
SMOOTH_MIN_RUN = 1                  
```

Valores buenos para **Madrid urbano/periurbano**. Adaptar según zona.

---

## TROUBLESHOOTING RÁPIDO

**Q: ¿edge_id=-1?**  
A: ⬆️ --max-dist 150→180. Si sigue, revisar cobertura red.

**Q: ¿Aceras/senderos?**  
A: Verificar `network_type="drive"`. Borrar `output/cached_networks`, regenerar.

**Q: ¿Muy lento?**  
A: ⬇️ SAMPLE_DENSITY 0.25→0.5. O ⬇️ --max-dist.

**Q: ¿No gira?**  
A: ⬇️ SWITCH_THRESHOLD 0.40→0.30. ⬇️ HISTORY_WINDOW 4→3.

**Q: ¿Parpadea?**  
A: ⬆️ SWITCH_THRESHOLD 0.40→0.50. ⬆️ HISTORY_BONUS 0.45→0.55.

---

## REFERENCIA FÓRMULAS

**Score:** `score = d²×(1-w) + dir×w` donde d=dist/max_dist, w=dir_weight, menor es mejor

**Pesos históricos:** `peso[i] = decay^(n-i)` donde n=HISTORY_WINDOW, decay=0.70

**Cambio aceptado si:** `(s_actual - s_mejor) / s_actual > SWITCH_THRESHOLD`

**Post-proceso:** votación vecinal en radio SMOOTH_RADIUS, elimina bloques ≤ SMOOTH_MIN_RUN

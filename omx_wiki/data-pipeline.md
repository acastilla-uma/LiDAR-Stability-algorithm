---
title: Data Pipeline
category: architecture
tags: [pipeline, data, gps, imu, map-matching]
---

# Data Pipeline

El pipeline convierte datos crudos DOBACK en CSVs enriquecidos listos para modelado y visualizacion.

## Etapa 1: GPS + estabilidad crudos

Entradas:

```text
Doback-Data/GPS/
Doback-Data/Stability/
```

Modulo:

```text
src/lidar_stability/parsers/batch_processor.py
```

Funciones importantes:

- `parse_gps_file`: lee GPS, extrae timestamp, lat/lon, altitud, HDOP, fix, satelites y velocidad.
- `parse_stability_file`: lee estabilidad/IMU.
- `load_outlier_filter_config`: fusiona defaults con `config/config.py` si existe.
- `detect_imu_outliers_rolling`: detecta outliers IMU con estadistica robusta.
- `match_by_timestamp`: une GPS e IMU por tolerancia temporal.
- `split_into_segments`: separa rutas por gaps espaciales.
- `process_all`: orquesta procesamiento batch.

Salida:

```text
Doback-Data/processed-data/*.csv
Doback-Data/processed-data/outliers/*
```

## Etapa 2: Map-matching

Modulo principal:

```text
src/lidar_stability/parsers/map_matching.py
```

Responsabilidades:

- Convertir lon/lat a UTM y viceversa.
- Cargar red desde GraphML local o construirla con OSMnx si esta disponible.
- Crear indice espacial.
- Proyectar puntos GPS sobre segmentos de carretera.
- Suavizar asignaciones de edge.
- Escribir CSVs map-matched.

Comando:

```bash
python src/lidar_stability/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched
```

Salida:

```text
Doback-Data/map-matched/*.csv
```

## Etapa 3: Enriquecimiento LiDAR

Modulo:

```text
src/lidar_stability/lidar/compute_route_terrain_features.py
```

Responsabilidades:

- Buscar tiles LAZ cercanos a cada punto.
- Extraer nubes locales.
- Construir DEM local.
- Calcular `phi_lidar`, `tri`, `ruggedness` y estadisticos de elevacion.
- Anadir constantes literales del dispositivo DOBACK.

Salida:

```text
Doback-Data/featured/*.csv
```

## Etapa 4: Ground truth y ML

Modulos:

- `src/lidar_stability/pipeline/ground_truth.py`
- `src/lidar_stability/ml/feature_engineering.py`
- `src/lidar_stability/ml/train_models_cli.py`
- `src/lidar_stability/ml/adaptive_hyperparam_search.py`

Salidas:

```text
output/models/*metrics.json
output/models/*models.joblib
output/models/*history.json
output/models/*leaderboard.*
```

## Etapa 5: Visualizacion

Modulos:

- `src/lidar_stability/visualization/cli.py`
- `src/lidar_stability/visualization/visualize_segment.py`
- `src/lidar_stability/visualization/map_builder.py`

Salidas:

```text
output/visualization/*.html
output/visualization/*.png
```


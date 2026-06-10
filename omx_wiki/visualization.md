---
title: Visualization
category: reference
tags: [visualization, maps, point-cloud, segments]
---

# Visualization

La visualizacion conecta resultados de modelos, segmentos featured y nubes LiDAR.

## Componentes

### `segment_ranking.py`

Lee metricas/predicciones de modelos y genera rankings:

- segmentos de impacto positivo
- segmentos de impacto negativo
- scores de error/impacto

### `segment_loader.py`

Carga:

- CSV featured de un segmento.
- tiles LAZ asociados.
- subconjuntos por rango de puntos.

### `point_cloud_processor.py`

Procesa nube de puntos:

- carga LAZ
- calcula bounds
- filtra por buffer alrededor de ruta
- decima puntos para que el HTML sea manejable

### `map_builder.py`

Genera visualizaciones:

- `create_segment_visualization_2d`
- `create_segment_visualization_3d`
- `create_segment_visualization`

Colorea rutas por SI y terreno por elevacion.

## CLI interactiva

```bash
python src/lidar_stability/visualization/cli.py \
  --model all-devices-no-imu \
  --top-n 5 \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --view-mode 3d
```

La CLI:

1. Ranking de segmentos por modelo.
2. Muestra opciones en terminal.
3. Permite elegir segmento.
4. Carga ruta y LiDAR.
5. Genera mapa en `output/visualization/`.

## Visualizar un segmento directo

```bash
python src/lidar_stability/visualization/visualize_segment.py \
  DOBACK024_20251007_seg28 \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --view-mode both
```

Opciones utiles:

- `--point-start`
- `--point-end`
- `--point-step`
- `--featured-dir`
- `--laz-dir`
- `--laz-source`
- `--output-dir`

## Comparar metricas de modelos

```bash
python src/lidar_stability/visualization/compare_models_metrics.py \
  --root output/models \
  --out output/model_metrics_figs
```

Genera barras para:

- `holdout_r2`
- `holdout_rmse`
- `generalization_gap`

## Scripts auxiliares

```bash
python scripts/visualization/visualize_models.py
```

Este script produce comparativas mas amplias por dispositivo/modelo y reportes.


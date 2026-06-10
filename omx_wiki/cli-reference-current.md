---
title: Current CLI Reference
category: reference
tags: [cli, commands, usage]
---

# Current CLI Reference

Esta pagina lista CLIs que existen en el arbol actual. Ejecutar siempre desde la raiz del repositorio.

## Procesamiento DOBACK

```bash
python src/lidar_stability/parsers/batch_processor.py --help
```

Uso tipico:

```bash
python src/lidar_stability/parsers/batch_processor.py \
  --data-dir Doback-Data \
  --output-dir Doback-Data/processed-data \
  --tolerance-seconds 1.0 \
  --max-gap-meters 1000
```

## Map-matching

```bash
python src/lidar_stability/parsers/map_matching.py --help
```

Uso tipico:

```bash
python src/lidar_stability/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched \
  --glob "*.csv"
```

Opciones relevantes:

- `--file`: procesa un solo CSV.
- `--network`: GraphML local.
- `--max-dist`: radio maximo de busqueda.
- `--dir-weight`: peso de direccion/heading.
- `--no-cache`: evita cache de red.

## Visualizador de rutas procesadas

```bash
python src/lidar_stability/parsers/route_visualizer.py --help
```

Uso tipico:

```bash
python src/lidar_stability/parsers/route_visualizer.py \
  DOBACK024_20251005 \
  --search-dir Doback-Data/processed-data \
  --output output/mapa_ruta_si.html
```

## Features de terreno

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py --help
```

Archivo individual:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
  --laz-dir LiDAR-Maps/cnig \
  --output Doback-Data/featured/DOBACK024_20251009_seg87.csv \
  --search-radius 100 \
  --dem-size 256 \
  --vehicle-track 2.48
```

Batch por DOBACK:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --doback DOBACK024 \
  --mapmatch-dir Doback-Data/map-matched \
  --featured-dir Doback-Data/featured \
  --laz-dir LiDAR-Maps/cnig
```

## Descarga de tiles CNIG

```bash
python src/lidar_stability/lidar/download_cnig_lidar_tiles.py --help
```

Uso tipico:

```bash
python src/lidar_stability/lidar/download_cnig_lidar_tiles.py \
  --tile-list output/cnig_missing_tiles_all_doback.txt \
  --output-dir LiDAR-Maps/cnig \
  --dry-run
```

## Auditoria de cobertura

```bash
python src/lidar_stability/pipeline/audit_pipeline_coverage.py --help
```

Uso tipico:

```bash
python src/lidar_stability/pipeline/audit_pipeline_coverage.py --simple
```

## Entrenamiento ML

Baseline simple:

```bash
python src/lidar_stability/ml/train_w_model.py --help
```

Entrenamiento multi-modelo:

```bash
python src/lidar_stability/ml/train_models_cli.py --help
```

Busqueda adaptativa:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py --help
```

Opciones importantes:

- `--target-unit {deg_s,rad_s}`: unidad fisica del target `gy`.
- `--cv-group-by source_file`: recomendado para artifacts con split agrupado en `train_models_cli.py`.
- `--run-config`: define corridas repetibles en JSON para `train_models_cli.py`.
- `--resume/--no-resume`: controla reanudacion de estudios Optuna en `adaptive_hyperparam_search.py`.

## Evaluacion PIML de SI

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability --help
```

Uso tipico con bundle compacto:

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability \
  --input-files Doback-Data/featured/DOBACK024_20251009_seg87.csv \
  --model-artifact output/models/w_model_models.joblib \
  --artifact-key rf \
  --output-csv output/stability/predictions.csv \
  --metrics-json output/stability/metrics.json \
  --group-column source_file
```

Opciones relevantes:

- `--model-artifact`: artifact `.joblib` o bundle compacto.
- `--artifact-key`: clave dentro del bundle, por ejemplo `rf`.
- `--vehicle-config`: YAML de parametros de vehiculo.
- `--si-column`: columna de SI medido.
- `--phi-lidar-column`: pendiente LiDAR en radianes.
- `--ay-column` y `--ay-unit`: aceleracion lateral y unidad.
- `--design-ay-m-s2`: aceleracion lateral de diseno si no hay columna.
- `--omega-target-unit`: unidad de `gy` medido para metricas de omega.
- `--benchmark-mode`: modo de desarrollo; no usar para metricas finales defensibles.

## Visualizacion de segmentos

CLI interactiva:

```bash
python src/lidar_stability/visualization/cli.py \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --view-mode 3d
```

Segmento concreto:

```bash
python src/lidar_stability/visualization/visualize_segment.py \
  DOBACK024_20251007_seg28 \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --view-mode both
```

## Scripts auxiliares

Analisis de outliers IMU:

```bash
python scripts/analysis/analyze_featured_imu_outliers_pie.py
```

Visualizacion comparativa de modelos:

```bash
python scripts/visualization/visualize_models.py
```

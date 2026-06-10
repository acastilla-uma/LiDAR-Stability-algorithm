---
title: Quick Start
category: reference
tags: [usage, install, quickstart, commands]
---

# Quick Start

## 1. Instalar dependencias

Desde la raiz del repositorio:

```bash
pip install -r requirements.txt
```

Para desarrollo y tests:

```bash
pip install -e . pytest
```

Si vas a usar busqueda adaptativa:

```bash
pip install optuna
```

## 2. Verificar entorno

```bash
python -m pytest
```

Resultado esperado en el estado actual de la wiki:

```text
58 passed, 16 skipped, 1 warning
```

Los skipped dependen de datos LiDAR reales o ficheros pesados que pueden no estar presentes.

## 3. Procesar datos crudos

```bash
python src/lidar_stability/parsers/batch_processor.py \
  --data-dir Doback-Data \
  --output-dir Doback-Data/processed-data
```

Esto produce CSVs segmentados y limpios.

## 4. Hacer map-matching

```bash
python src/lidar_stability/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched
```

Tambien se puede procesar un unico archivo con `--file`.

## 5. Enriquecer con terreno LiDAR

Modo archivo:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
  --laz-dir LiDAR-Maps/cnig \
  --output Doback-Data/featured/DOBACK024_20251009_seg87.csv
```

Modo batch por dispositivo:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --doback DOBACK024 \
  --mapmatch-dir Doback-Data/map-matched \
  --featured-dir Doback-Data/featured \
  --laz-dir LiDAR-Maps/cnig
```

## 6. Entrenar modelos

Entrenamiento compacto de modelos:

```bash
python src/lidar_stability/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --models rf extra_trees gbr \
  --cv-group-by source_file \
  --target-unit deg_s \
  --output-dir output/models \
  --prefix w_model
```

Busqueda adaptativa con Optuna:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --model rf \
  --max-trials 80 \
  --target-r2 0.70 \
  --target-unit deg_s \
  --output-dir output/models
```

## 7. Evaluar SI final con PIML

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability \
  --input-files Doback-Data/featured/DOBACK024_20251009_seg87.csv \
  --model-artifact output/models/w_model_models.joblib \
  --artifact-key rf \
  --output-csv output/stability/predictions.csv \
  --metrics-json output/stability/metrics.json \
  --group-column source_file
```

El modelo predice `omega`; la capa fisica calcula `si_static_lidar_risk`, `si_dynamic_omega_risk`, `si_risk_total` y `si_pred`.

## 8. Visualizar segmentos

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

Ver detalles en [[usage-guide]], [[cli-reference-current]], [[visualization]], [[ml-training]] y [[piml-stability-pipeline]].

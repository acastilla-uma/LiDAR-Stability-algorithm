---
title: Usage Guide
category: reference
tags: [usage, guide, cli, pipeline, piml]
---

# Usage Guide

Esta guia recorre el uso completo del repositorio desde datos crudos DOBACK hasta evaluacion PIML del indice de estabilidad.

Ejecutar los comandos desde la raiz del repositorio.

## 1. Preparar entorno

Instalacion minima:

```bash
pip install -r requirements.txt
```

Modo desarrollo:

```bash
pip install -e . pytest
```

Busqueda adaptativa:

```bash
pip install optuna
```

Comprobacion:

```bash
python -m pytest -q
```

Estado verificado de esta wiki:

```text
58 passed, 16 skipped, 1 warning
```

Los skipped suelen depender de datos LiDAR reales o ficheros pesados no incluidos en el repo.

## 2. Estructura esperada de datos

Entradas principales:

```text
Doback-Data/GPS/*.txt
Doback-Data/Stability/*.txt
LiDAR-Maps/cnig/*.laz
LiDAR-Maps/geo-mad/*.tif
```

Salidas principales:

```text
Doback-Data/processed-data/*.csv
Doback-Data/map-matched/*.csv
Doback-Data/featured/*.csv
output/models/*
output/stability/*
output/visualization/*
```

Los directorios de datos pesados no deben versionarse.

## 3. Procesar datos crudos DOBACK

```bash
python src/lidar_stability/parsers/batch_processor.py \
  --data-dir Doback-Data \
  --output-dir Doback-Data/processed-data \
  --tolerance-seconds 1.0 \
  --max-gap-meters 1000
```

Resultado esperado:

- CSVs limpios y segmentados.
- Columnas GPS/IMU sincronizadas.
- Auditoria de outliers cuando aplique.

## 4. Map-matching

Batch:

```bash
python src/lidar_stability/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched \
  --glob "*.csv"
```

Archivo individual:

```bash
python src/lidar_stability/parsers/map_matching.py \
  --file Doback-Data/processed-data/DOBACK024_20251009_seg87.csv \
  --output Doback-Data/map-matched
```

Opciones utiles:

- `--network`: GraphML local.
- `--max-dist`: distancia maxima al segmento vial.
- `--dir-weight`: peso de direccion.
- `--no-cache`: evita cache de red.

## 5. Extraer features LiDAR

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

Batch por dispositivo:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --doback DOBACK024 \
  --mapmatch-dir Doback-Data/map-matched \
  --featured-dir Doback-Data/featured \
  --laz-dir LiDAR-Maps/cnig
```

Columnas clave esperadas en featured:

- `roll`, `pitch`, `ax`, `ay`, `az`
- `speed_kmh`
- `phi_lidar` en radianes
- `tri`
- `ruggedness`
- `gy`
- `si` si existe SI medido en la base de datos

## 6. Entrenar modelo de omega

El modelo aprende `gy`/omega. No debe aprender `si` final.

Entrenamiento multi-modelo recomendado:

```bash
python src/lidar_stability/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --models rf extra_trees gbr \
  --cv-group-by source_file \
  --target-unit deg_s \
  --n-splits 5 \
  --output-dir output/models \
  --prefix w_model
```

Salida compacta por defecto:

```text
output/models/w_model_models.joblib
output/models/w_model_metrics.json
output/models/w_model_leaderboard.json
```

El artifact guarda:

- `feature_columns` y `feature_order`
- `feature_units`
- `target_unit`
- `prediction_unit`
- `split_metadata`
- `sklearn_version`

## 7. Busqueda adaptativa con Optuna

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --model rf \
  --target-r2 0.70 \
  --max-trials 80 \
  --patience 25 \
  --target-unit deg_s \
  --output-dir output/models \
  --prefix adaptive_w_model
```

Produce:

- mejor artifact `.joblib`
- historial JSON
- leaderboard CSV/JSON
- predicciones holdout
- reporte markdown

La busqueda adaptativa usa holdout agrupado por source file y registra `split_metadata.kind = grouped`.

## 8. Evaluar SI final con pipeline PIML

El evaluador carga CSVs featured y un modelo de omega, predice `omega_rad_s`, calcula fisica y compara contra `si` medido.

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability \
  --input-files Doback-Data/featured/DOBACK024_20251009_seg87.csv \
  --model-artifact output/models/w_model_models.joblib \
  --artifact-key rf \
  --output-csv output/stability/DOBACK024_20251009_seg87_predictions.csv \
  --metrics-json output/stability/DOBACK024_20251009_seg87_metrics.json \
  --group-column source_file
```

Si se usa un artifact no compacto:

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability \
  --input-files Doback-Data/featured/DOBACK024_20251009_seg87.csv \
  --model-artifact output/models/w_model_rf.joblib \
  --output-csv output/stability/predictions.csv \
  --metrics-json output/stability/metrics.json
```

Opciones importantes:

- `--artifact-key`: modelo dentro de bundle compacto.
- `--vehicle-config`: YAML con geometria del vehiculo.
- `--si-column`: columna de SI medido, por defecto `si`.
- `--phi-lidar-column`: pendiente transversal LiDAR en radianes.
- `--ay-column`: aceleracion lateral.
- `--ay-unit`: `m_s2`, `cm_s2` o `g`.
- `--design-ay-m-s2`: valor de diseno si no hay columna `ay`.
- `--omega-target-unit`: unidad de la columna medida `gy` para metricas de omega.
- `--benchmark-mode`: solo para baselines; permite artifacts leakage-prone y metricas no finales.

## 9. Interpretar salidas PIML

CSV de predicciones:

- `omega_pred_raw`
- `omega_pred_unit`
- `omega_pred_rad_s`
- `omega_true_rad_s` si existe `gy`
- `phi_crit_rad`
- `omega_crit_rad_s`
- `si_static_lidar_risk`
- `si_static_lidar_margin`
- `si_dynamic_omega_risk`
- `si_risk_total`
- `si_pred`
- `si_true` y `si_error` si existe SI medido

JSON de metricas:

- `omega`: RMSE, MAE, R2 y residuos de omega.
- `static_only_si`: baseline estatico transformado a margen.
- `final_si`: SI final contra SI medido.
- `physics`: unidades, parametros, `phi_crit`, resumen de `omega_crit`.
- `monotonicity`: violaciones al aumentar pendiente u omega.
- `si_band_confusion`: tabla por bandas `danger`, `warning`, `safe`.

Semantica final:

- `SI = 1`: maxima estabilidad.
- Menor SI: menor estabilidad.
- Columnas `*_risk`: aumentan con inestabilidad.
- Columnas `*_margin`: aumentan con estabilidad.

## 10. Visualizacion

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

Comparativa de modelos:

```bash
python scripts/visualization/visualize_models.py
```

## 11. Calidad y troubleshooting

Tests completos:

```bash
python -m pytest -q
```

Tests focales del pipeline PIML:

```bash
python -m pytest tests/test_stability_pipeline.py tests/test_sprint5.py -q
```

Errores comunes:

- `No valid rows after numeric feature conversion`: faltan features o hay NaNs en columnas del modelo.
- `Artifact feature_columns contain leakage-prone columns`: el modelo usa `si`, `delta_si` u otra columna derivada del objetivo final.
- `Final SI metrics require grouped artifact split metadata`: el artifact no demuestra split agrupado; usar `--cv-group-by source_file` al entrenar o `--benchmark-mode` solo para desarrollo.
- `Input dataframe is missing 'ay'`: pasar `--ay-column` correcto o `--design-ay-m-s2`.
- `ModuleNotFoundError: lidar_stability`: usar `PYTHONPATH=src` al ejecutar con `python -m`.

Ver tambien: [[quick-start]], [[cli-reference-current]], [[piml-stability-pipeline]], [[ml-training]], [[testing-and-quality]].

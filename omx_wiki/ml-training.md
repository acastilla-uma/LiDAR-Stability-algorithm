---
title: ML Training
category: architecture
tags: [ml, training, optuna, models]
---

# ML Training

El paquete `src/lidar_stability/ml` entrena modelos sobre CSVs featured.

## Feature engineering

Archivo:

```text
src/lidar_stability/ml/feature_engineering.py
```

Funcion principal:

```python
build_w_training_dataset(df, feature_columns=None, target_column=None)
```

Devuelve:

- `X`: dataframe numerico con features.
- `y`: serie target.
- `used_features`: columnas realmente usadas.
- `clean_df`: dataframe filtrado para filas validas.

Features por defecto:

```text
roll, pitch, ax, ay, az, speed_kmh, phi_lidar, tri, ruggedness
```

Target por defecto:

```text
gy
```

Por defecto `gy` se registra en artifacts como `deg_s`; el evaluador PIML lo convierte a `omega_rad_s` antes de aplicar la fisica.

Las constantes de dispositivo solo entran si se pasan explicitamente en `--feature-columns`.

## Entrenamiento multi-modelo

Archivo:

```text
src/lidar_stability/ml/train_models_cli.py
```

Modelos soportados:

- `rf`: RandomForestRegressor.
- `extra_trees`: ExtraTreesRegressor.
- `gbr`: GradientBoostingRegressor.

Comando base:

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

Salidas:

- bundle `.joblib`
- metricas `.json`
- contexto de entrenamiento
- `target_unit`, `prediction_unit`, `feature_order`, `feature_units`, `split_metadata` y `sklearn_version`
- folds y metricas por corrida

## Configurar varias corridas

`--run-config` acepta JSON repetible:

```bash
python src/lidar_stability/ml/train_models_cli.py \
  --run-config '{"model":"rf","run_id":"rf_350","rf_n_estimators":350,"rf_min_samples_leaf":3}' \
  --run-config '{"model":"gbr","run_id":"gbr_300","gbr_n_estimators":300}'
```

## Busqueda adaptativa con Optuna

Archivo:

```text
src/lidar_stability/ml/adaptive_hyperparam_search.py
```

Comando:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --model rf \
  --target-r2 0.70 \
  --max-trials 80 \
  --patience 25 \
  --output-dir output/models \
  --prefix adaptive_w_model
```

Conceptos:

- Usa train/holdout por grupos de source file.
- Ejecuta CV en train.
- Evalua holdout por trial.
- Guarda estudio SQLite si `--resume` esta activo.
- Exporta `history`, `leaderboard`, predicciones holdout y mejor modelo.

## Leaderboard

Archivo:

```text
src/lidar_stability/ml/plot_models_leaderboard.py
```

Funcion:

```python
load_metrics(root, enrich_from_bundle=True)
```

Lee `*metrics.json`, `*history.json` y `*leaderboard.json` para construir un dataframe ordenable.

Ver tambien: [[piml-stability-pipeline]], [[configuration-and-data]], [[testing-and-quality]].

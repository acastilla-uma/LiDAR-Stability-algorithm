# Guia de entrenamiento y evaluacion (Sprint 5)

Esta guia resume como entrenar, evaluar y guardar modelos para predecir `w` (omega) usando datos `featured`.

Incluye dos opciones:
- Notebook documentado para exploracion y analisis visual.
- Script CLI para entrenamiento reproducible y seleccion flexible de datos.

## 1) Notebook de documentacion

Notebook principal:
- `var-analysis/guia_entrenamiento_evaluacion_sprint5.ipynb`

Contenido del notebook:
- Carga de CSVs desde `Doback-Data/featured`.
- Construccion del dataset supervisado con `build_w_training_dataset`.
- Entrenamiento con K-Fold (`RandomForestRegressor`).
- Metricas de evaluacion: `RMSE`, `MAE`, `R2`.
- Graficas:
  - Distribucion del target.
  - Predicho vs real.
  - Histograma de residuales.
  - Residuales vs prediccion.
  - Metricas por fold.
  - Importancia de variables.
- Guardado de artefactos:
  - Modelo `.joblib`.
  - Metricas `.json`.

## 2) Script CLI para entrenar y guardar modelos

Script:
- `Scripts/ml/train_models_cli.py`

Permite:
- Seleccionar datos por glob, lista de archivos, filtros por nombre y limite de archivos.
- Filtrar filas por condicion (`pandas.query`).
- Entrenar uno o varios modelos.
- Guardar modelos, metricas y leaderboard comparativo.

### Modelos soportados

- `rf` (Random Forest)
- `extra_trees` (Extra Trees)
- `gbr` (Gradient Boosting Regressor)

## 3) Ejemplos de uso

### 3.1 Entrenar un solo modelo con todos los featured

```bash
python Scripts/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --models rf \
  --n-splits 5 \
  --output-dir "output/results/models" \
  --prefix "w_prod"
```

### 3.2 Entrenar y comparar varios modelos

```bash
python Scripts/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK024_20250929_seg*.csv" \
  --models rf extra_trees gbr \
  --n-splits 5 \
  --output-dir "output/results/models_compare" \
  --prefix "w_compare"
```

### 3.3 Elegir subconjunto de archivos y filtrar filas

```bash
python Scripts/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK*.csv" \
  --contains DOBACK024 20250929 \
  --max-files 3 \
  --shuffle-files \
  --query "speed_kmh > 10 and abs(roll) < 15" \
  --models rf \
  --output-dir "output/results/models_filtered" \
  --prefix "w_filtered"
```

### 3.4 Entrenar con archivo especifico (smoke test)

```bash
python Scripts/ml/train_models_cli.py \
  --input-glob "Doback-Data/featured/DOBACK024_20250929_seg13.csv" \
  --models rf extra_trees \
  --n-splits 3 \
  --output-dir "output/results/models_smoke" \
  --prefix "w_smoke"
```

## 4) Parametros importantes del CLI

- `--input-glob`: uno o varios patrones de CSV.
- `--input-files`: rutas explicitas de CSV.
- `--contains`: texto que debe aparecer en el nombre de archivo.
- `--max-files`: limita cantidad de archivos seleccionados.
- `--shuffle-files`: mezcla orden de archivos antes de cortar con `--max-files`.
- `--query`: filtro de filas tipo pandas (`"speed_kmh > 10"`).
- `--models`: lista de modelos a entrenar (`rf extra_trees gbr`).
- `--n-splits`: folds para validacion cruzada.
- `--target-column`: target explicito (si no, autodeteccion).
- `--feature-columns`: lista de columnas de entrada (opcional).
- `--output-dir`: carpeta de salida de artefactos.
- `--prefix`: prefijo para nombres de archivo generados.

## 5) Archivos de salida

Por cada modelo entrenado:
- `<output-dir>/<prefix>_<model>.joblib`
- `<output-dir>/<prefix>_<model>_metrics.json`

Comparativa global:
- `<output-dir>/<prefix>_leaderboard.json`

Ejemplo real (smoke):
- `output/results/models_smoke/w_smoke_rf.joblib`
- `output/results/models_smoke/w_smoke_rf_metrics.json`
- `output/results/models_smoke/w_smoke_extra_trees.joblib`
- `output/results/models_smoke/w_smoke_extra_trees_metrics.json`
- `output/results/models_smoke/w_smoke_leaderboard.json`

## 6) Metricas e interpretacion

- `RMSE`: penaliza mas errores grandes (menor es mejor).
- `MAE`: error medio absoluto (menor es mejor).
- `R2`: varianza explicada (mayor es mejor, ideal cercano a 1).

Se reportan:
- Media y desviacion por K-Fold.
- Metricas por fold en el JSON.

## 7) Flujo recomendado en proyecto

1. Explorar y ajustar en notebook (`var-analysis/guia_entrenamiento_evaluacion_sprint5.ipynb`).
2. Ejecutar entrenamiento reproducible con CLI (`Scripts/ml/train_models_cli.py`).
3. Revisar `leaderboard` y elegir modelo.
4. Usar artefacto seleccionado para integracion en pipeline de SI dinamico.

---
title: Testing And Quality
category: reference
tags: [tests, quality, pytest, maintenance]
---

# Testing And Quality

## Comando principal

```bash
python -m pytest
```

Estado registrado en la ultima actualizacion de esta wiki:

```text
58 passed, 16 skipped, 1 warning
```

## Que cubren los tests

### `tests/test_sprint1.py`

- batch processing
- route visualizer
- `StabilityEngine`
- ground truth basico

### `tests/test_sprint2.py`

- EKF
- time sync GPS/IMU
- merge GPS/IMU
- batch helpers de EKF

### `tests/test_sprint3.py`

- LAZ reader
- TIF reader
- TerrainProvider
- TerrainFeatureExtractor
- tests con datos reales se saltan si no hay ficheros LiDAR

### `tests/test_sprint5.py`

- enhanced ground truth
- feature engineering para `gy`

### `tests/test_stability_pipeline.py`

- formulas fisicas de riesgo y margen SI
- evaluador PIML con artifact sintetico
- guardas de leakage
- guardas de metadata agrupada para metricas finales
- conversion independiente entre unidad de prediccion y unidad de `gy` medido

### `tests/test_adaptive_hyperparam_search_optuna.py`

- espacios de hiperparametros
- CV grouped por source file
- serializacion de historial Optuna
- compatibilidad de leaderboard
- smoke de training CLI

### `tests/test_device_constants_enrichment.py`

- extraccion de constantes por filename
- enriquecimiento de route features con constantes
- uso explicito de constantes en ML

## Checks utiles

Compilar Python:

```bash
python -m compileall src scripts
```

Estado de git:

```bash
git status --short
```

Buscar referencias a archivos inexistentes:

```bash
rg -n "run_full_pipeline|visualize_route_lidar|build_enhanced_ground_truth.py|compare_stability_csv"
```

## Riesgos conocidos

- Algunas docs historicas pueden mencionar scripts que ya no existen; la referencia vigente es [[usage-guide]] y [[cli-reference-current]].
- Hay broad `except Exception` en varios CLIs; algunos son fallbacks razonables de IO, pero conviene revisarlos por modulo antes de grandes refactors.
- La etapa LiDAR puede ser lenta y depende de datos pesados locales.
- `ruff` no estaba instalado en el entorno usado para la ultima limpieza.

## Reglas de mantenimiento

- Mantener salidas generadas en `output/`.
- No escribir artefactos en `src/`.
- No versionar `tmp/`, cache de matplotlib ni datos LiDAR.
- Antes de refactors, correr tests focalizados y luego suite completa.
- Si se agrega una CLI nueva, actualizar [[cli-reference-current]] y [[module-map]].

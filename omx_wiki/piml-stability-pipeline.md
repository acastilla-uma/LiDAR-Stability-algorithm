---
title: PIML Stability Pipeline
category: architecture
tags: [piml, stability, physics, evaluator, si]
---

# PIML Stability Pipeline

El pipeline PIML separa aprendizaje y fisica:

1. El modelo ML predice solo `omega_rad_s`.
2. La capa fisica calcula el termino estatico con `phi_lidar`.
3. La capa fisica calcula el termino dinamico con `omega_pred_rad_s`.
4. El evaluador suma riesgos, transforma a margen de estabilidad y compara contra el SI medido.

## Semantica SI

El SI final es un margen:

- `SI = 1`: maxima estabilidad.
- Valores menores: menor estabilidad.
- Las columnas `*_risk` suben con la inestabilidad.
- La salida final usa `si_pred = clip(1 - si_risk_total, 0, 1)`.

## Formulas

El motor esta en:

```text
src/lidar_stability/physics/stability_engine.py
```

Terminos principales:

```text
phi_crit_rad = atan((s_m / 2) / h_m)
si_static_lidar_risk = k1 * abs(phi_lidar_rad) / phi_crit_rad
omega_crit_rad_s = sqrt(max(coeff * s_m * abs(ay_m_s2) * correction_factor / 4, eps))
si_dynamic_omega_risk = k2 * (abs(omega_pred_rad_s) / omega_crit_rad_s) ** 2
si_risk_total = si_static_lidar_risk + si_dynamic_omega_risk
si_pred = clip(1 - si_risk_total, 0, 1)
```

Unidades esperadas:

- `phi_lidar`: radianes.
- `omega_pred_rad_s`: rad/s.
- `ay_m_s2`: m/s^2 tras conversion desde la columna configurada.
- `track_width_m` y `cg_height_m`: metros.

## Evaluador

Archivo:

```text
src/lidar_stability/pipeline/evaluate_stability.py
```

Uso tipico:

```bash
PYTHONPATH=src python -m lidar_stability.pipeline.evaluate_stability \
  --input-files Doback-Data/featured/DOBACK_01.csv \
  --model-artifact output/models/w_model_models.joblib \
  --artifact-key rf \
  --output-csv output/stability/predictions.csv \
  --metrics-json output/stability/metrics.json \
  --group-column source_file
```

Salidas del CSV:

- `omega_pred_rad_s`
- `omega_true_rad_s` si existe `gy`
- `si_static_lidar_risk`
- `si_static_lidar_margin`
- `si_dynamic_omega_risk`
- `si_risk_total`
- `si_pred`
- `si_true` y `si_error` si existe SI medido

Salidas del JSON:

- `omega`: RMSE, MAE, R2 y residuos de omega.
- `static_only_si`: baseline estatico como margen.
- `final_si`: metrica final contra SI medido.
- `monotonicity`: violaciones si al aumentar pendiente u omega sube el SI.
- `physics`: unidades, parametros y resumen de `omega_crit`.
- `si_band_confusion`: tabla por bandas `danger`, `warning`, `safe`.

## Guardas contra leakage

El evaluador rechaza artifacts cuyas `feature_columns` incluyan columnas derivadas de SI medido, por ejemplo:

```text
si, si_mcu, si_real, si_pred, delta_si, si_dynamic_obs
```

Para metricas finales contra SI medido tambien exige evidencia de evaluacion agrupada:

- artifact entrenado con split agrupado, por ejemplo `--cv-group-by source_file` o `split_metadata.kind = grouped`.

`--group-column` solo anade resumen descriptivo por grupos cuando la metrica final ya esta permitida por el artifact. No convierte un modelo entrenado con split por filas en evidencia de generalizacion.

`--benchmark-mode` existe solo para baselines de desarrollo y deja reflejado el modo en las metricas.

## Columnas legacy y diagnosticas

Las columnas nuevas deben preferirse siempre:

- `*_risk`: aumenta con inestabilidad.
- `*_margin`: aumenta con estabilidad.
- `si_pred`: margen final.

Por compatibilidad, algunos builders aun emiten columnas legacy como `si_static`, `si_static_lidar` o `si_static_fused`; esas columnas representan riesgo/utilizacion historica, no margen final.

Las columnas `si_dynamic_obs`, `si_pred_obs_w` y `delta_si*` son diagnosticas u observadas. Pueden ayudar a inspeccionar residuos, pero son leakage-prone y no deben entrar como features del modelo.

## Entrenamiento compatible

Los trainers siguen usando `gy` como target por defecto. Los artifacts nuevos guardan:

- `target_unit`
- `prediction_unit`
- `feature_order`
- `feature_units`
- `split_metadata`
- `sklearn_version`

Esto permite que el evaluador convierta predicciones crudas a `omega_rad_s` antes de aplicar la fisica.

Ver tambien: [[ml-training]], [[data-pipeline]], [[testing-and-quality]].

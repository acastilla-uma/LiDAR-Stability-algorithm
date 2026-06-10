---
title: Home
category: reference
tags: [overview, lidar, stability, doback]
---

# Home

`LiDAR-Stability-algorithm` es un pipeline Python para procesar rutas DOBACK con datos GPS, datos de estabilidad/IMU y terreno LiDAR. El objetivo es producir datos enriquecidos por segmento y entrenar modelos que relacionen estabilidad, dinamica del vehiculo y caracteristicas de terreno.

## Que hace

- Lee ficheros crudos de GPS y estabilidad DOBACK.
- Limpia outliers y conserva auditoria de filas descartadas.
- Une GPS e IMU por timestamp.
- Divide rutas en segmentos manejables.
- Hace map-matching contra una red viaria.
- Enriquece puntos de ruta con features LiDAR: pendiente transversal, TRI, ruggedness y estadisticos de elevacion.
- Calcula ground truth de estabilidad estatica/dinamica.
- Entrena modelos para predecir `gy`/omega.
- Evalua SI final con arquitectura PIML: omega por ML y SI por capa fisica.
- Genera visualizaciones 2D/3D de segmentos y nubes de puntos.

## Paquete principal

Todo el codigo de aplicacion vive bajo:

```text
src/lidar_stability/
```

Subpaquetes principales:

- `parsers`: ingestion, limpieza, matching temporal y map-matching.
- `lidar`: lectura LAZ/TIF y extraccion de features de terreno.
- `config`: configuracion por dispositivo DOBACK.
- `physics`: motor de estabilidad estatica.
- `pipeline`: ground truth, evaluador PIML y auditorias de cobertura.
- `ekf`: helpers de sincronizacion y filtro EKF.
- `ml`: feature engineering y entrenamiento.
- `visualization`: ranking, carga de segmentos, point clouds y mapas.

## Entradas y salidas esperadas

Entradas principales:

- `Doback-Data/GPS/*.txt`
- `Doback-Data/Stability/*.txt`
- `LiDAR-Maps/cnig/*.laz`
- Opcionalmente `LiDAR-Maps/geo-mad/*.tif`

Salidas principales:

- `Doback-Data/processed-data/*.csv`
- `Doback-Data/map-matched/*.csv`
- `Doback-Data/featured/*.csv`
- `output/models/*`
- `output/visualization/*`

Ver tambien: [[quick-start]], [[usage-guide]], [[data-pipeline]], [[architecture]], [[piml-stability-pipeline]].

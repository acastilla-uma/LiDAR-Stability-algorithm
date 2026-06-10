---
title: Configuration And Data
category: reference
tags: [configuration, data, devices, doback]
---

# Configuration And Data

## Layout de datos

```text
Doback-Data/
  GPS/
  Stability/
  processed-data/
    outliers/
  map-matched/
  featured/

LiDAR-Maps/
  cnig/
  geo-mad/

output/
  models/
  visualization/
```

## Datos ignorados por git

`.gitignore` ignora:

- `Doback-Data/`
- `LiDAR-Maps/`
- `cache/`
- `tmp/`
- `output/`
- formatos pesados: `*.laz`, `*.las`, `*.tif`, `*.tiff`

Esto es intencional: los datos LiDAR y salidas pueden ser enormes.

## Configuracion global

Archivo:

```text
config/config.py
```

Puede sobreescribir:

- validacion GPS (`GPS_VALIDATION`)
- filtros de outliers (`OUTLIER_FILTER_CONFIG`)
- parametros de visualizacion/mapa.

`batch_processor.py` carga estos overrides si existen y si tienen forma de diccionario.

## Configuracion por dispositivo

Directorio:

```text
src/lidar_stability/config/devices/
```

Archivos actuales:

- `doback-23.yaml`
- `doback-24.yaml`
- `doback-27.yaml`
- `doback-28.yaml`

Campos importantes:

- `device_id`
- `stability_model.k1`
- `stability_model.k2`
- `stability_model.d1_m`
- `stability_model.s_mm`
- `stability_model.coeff`
- `stability_model.alphav`

## Como se resuelve un dispositivo

`DeviceRegistry.get_device_from_filename` busca patrones como:

```text
DOBACK024_20251001_seg3.csv -> 24
GPS_DOBACK027_20250929.txt -> 27
ESTABILIDAD_DOBACK023_20251012.txt -> 23
```

Luego `device_constants.py` usa ese ID para insertar constantes literales en datasets featured.

## Columnas importantes

GPS/procesado:

- `timestamp`
- `lat`
- `lon`
- `x_utm`
- `y_utm`
- `speed_kmh`

IMU/estabilidad:

- `roll`
- `pitch`
- `yaw`
- `ax`, `ay`, `az`
- `gx`, `gy`, `gz`
- `si`

Featured:

- columnas anteriores
- `phi_lidar`
- `phi_lidar_deg`
- `tri`
- `ruggedness`
- `z_min`, `z_max`, `z_mean`, `z_std`, `z_range`
- constantes de dispositivo: `k1`, `k2`, `k4_mm`, `d1_m`, `coeff`, `s_mm`, `alphav`

Ver tambien: [[data-pipeline]], [[terrain-lidar]], [[ml-training]].


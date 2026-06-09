# Segment Visualization Guide

Esta guia explica como usar los scripts de visualizacion de segmentos LiDAR y que hace cada argumento.

## Scripts incluidos

### 1. CLI interactivo

Archivo: `src/lidar_stability/visualization/cli.py`

Uso basico:

```bash
python -m src.lidar_stability.visualization.cli --buffer-radius 5 --decimation-ratio 0.1
```

Este script calcula el ranking de segmentos y luego te pide un `segment_id` para visualizarlo.

### 2. Visualizacion de un segmento concreto

Archivo: `src/lidar_stability/visualization/visualize_segment.py`

Uso basico:

```bash
python -m src.lidar_stability.visualization.visualize_segment DOBACK024_20251007_seg28 --buffer-radius 5 --decimation-ratio 0.1
```

Este script no es interactivo: recibe un segmento concreto y genera el mapa directamente.

## Argumentos comunes

| Argumento | Tipo | Obligatorio | Descripcion |
| --- | --- | --- | --- |
| `--buffer-radius` | float | Si | Radio en metros para filtrar la nube LiDAR cerca de la ruta. Debe ser mayor que 0. |
| `--decimation-ratio` | float | Si | Fraccion de puntos que se conservan tras la reduccion. Debe estar en el rango `(0, 1]`. |
| `--view-mode` | string | No | Tipo de salida: `2d`, `3d` o `both`. Por defecto es `3d`. |
| `--point-start` | int | No | Indice inicial de la ruta que se visualiza. Por defecto `0`. |
| `--point-end` | int | No | Indice final exclusivo de la ruta. Si no se indica, se usa hasta el final. |
| `--point-step` | int | No | Muestra un punto cada N puntos del tramo seleccionado. Por defecto `1`. |
| `--featured-dir` | Path | No | Carpeta donde estan los CSV featured. Si no se pasa, se autodetecta. |
| `--laz-dir` | Path | No | Carpeta base de LAZ de referencia. Se usa como base para `cnig` cuando aplica. |
| `--laz-source` | string | No | Fuerza la fuente de LAZ: `geo-mad`, `cnig` o `both`. Por defecto `both`. |
| `--output-dir` | Path | No | Carpeta donde se guardan los HTML generados. |

## Argumentos del CLI interactivo

### `src/lidar_stability/visualization/cli.py`

| Argumento | Tipo | Obligatorio | Descripcion |
| --- | --- | --- | --- |
| `--model` | string | No | Nombre del directorio del modelo. Por defecto `all-devices-no-imu`. |
| `--top-n` | int | No | Numero de segmentos positivos y negativos a mostrar. Por defecto `5`. |
| `--buffer-radius` | float | Si | Radio de busqueda de puntos LiDAR alrededor de la ruta. |
| `--decimation-ratio` | float | Si | Fraccion de puntos conservados en la nube procesada. |
| `--view-mode` | string | No | Salida `2d`, `3d` o `both`. |
| `--point-start` | int | No | Primer punto de ruta a usar. |
| `--point-end` | int | No | Ultimo punto exclusivo de la ruta a usar. |
| `--point-step` | int | No | Paso de muestreo de la ruta. |
| `--models-dir` | Path | No | Ruta a `output/models/extra_trees` o carpeta equivalente. |
| `--featured-dir` | Path | No | Ruta a `Doback-Data/featured`. |
| `--laz-dir` | Path | No | Ruta base de LAZ. Por defecto apunta a `LiDAR-Maps/cnig`. |
| `--laz-source` | string | No | Fuerza el origen de LAZ: `geo-mad`, `cnig` o `both`. |
| `--output-dir` | Path | No | Carpeta destino para los mapas HTML. |

## Argumento posicional del script no interactivo

### `src/lidar_stability/visualization/visualize_segment.py`

| Argumento | Tipo | Obligatorio | Descripcion |
| --- | --- | --- | --- |
| `segment_id` | string o ruta CSV | Si | Identificador del segmento o ruta completa al CSV. Si pasas una ruta, el script la usa directamente. |

## Como elige la nube LiDAR

El orden de busqueda es:

1. `geo-mad` si se selecciona con `--laz-source geo-mad` o `--laz-source both`.
2. `cnig` si se selecciona con `--laz-source cnig` o `--laz-source both`.

Si solo quieres usar una base de datos, usa `--laz-source geo-mad` o `--laz-source cnig`.

## Ejemplos practicos

### Visualizar un segmento de `geo-mad`

```bash
python -m src.lidar_stability.visualization.visualize_segment \
  DOBACK028_20251115_seg47 \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --laz-source geo-mad
```

### Visualizar un CSV completo de `filtered_featured_geomad`

```bash
python -m src.lidar_stability.visualization.visualize_segment \
  /users/acastilla/Lidar-algority/LiDAR-Stability-algorithm/Doback-Data/filtered_featured_geomad/DOBACK028_20251115_seg47.csv \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --laz-source geo-mad
```

### Forzar CNIG aunque exista geo-mad

```bash
python -m src.lidar_stability.visualization.visualize_segment \
  DOBACK028_20251115_seg47 \
  --buffer-radius 5 \
  --decimation-ratio 0.1 \
  --laz-source cnig
```

## Requisitos de entrada

- El CSV del segmento debe incluir al menos estas columnas: `lat`, `lon`, `x_utm`, `y_utm`, `si`.
- Para visualizacion 3D o 2D, el script intentara cargar uno o varios archivos `.laz` compatibles.
- Si no encuentra datos LiDAR en la fuente elegida, fallara con un error explicito.

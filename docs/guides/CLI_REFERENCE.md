# CLI Reference

Esta guia documenta scripts CLI del proyecto que no tenian una guia dedicada o cuya documentacion estaba dispersa.

## Convenciones

- Ejecutar desde la raiz del repositorio.
- Todas las rutas de codigo estan bajo `src/lidar_stability`.
- Datos:
  - `Doback-Data/processed-data`
  - `Doback-Data/map-matched`
  - `Doback-Data/featured`

## Parsers

### Batch processor

Archivo: `src/lidar_stability/parsers/batch_processor.py`

Uso:

```bash
python src/lidar_stability/parsers/batch_processor.py --help
```

Funcion:
- Une GPS + estabilidad por tiempo.
- Filtra anomalias y segmenta rutas.
- Escribe CSVs procesados en `processed-data`.

### Route visualizer

Archivo: `src/lidar_stability/parsers/route_visualizer.py`

Uso:

```bash
python src/lidar_stability/parsers/route_visualizer.py --help
```

Funcion:
- Visualiza rutas procesadas sobre mapa.
- Soporta base name o lista de CSVs.

## Pipeline

### Full pipeline

Archivo: `src/lidar_stability/pipeline/run_full_pipeline.py`

Uso:

```bash
python src/lidar_stability/pipeline/run_full_pipeline.py --help
```

Funcion:
- Orquesta etapas de procesamiento, map-matching, features y visualizacion.

### Audit coverage

Archivo: `src/lidar_stability/pipeline/audit_pipeline_coverage.py`

Uso:

```bash
python src/lidar_stability/pipeline/audit_pipeline_coverage.py --simple
```

Funcion:
- Audita cobertura de rutas por etapa.
- Modo simple por dispositivo DOBACK.

### Build enhanced ground truth

Archivo: `src/lidar_stability/pipeline/build_enhanced_ground_truth.py`

Uso:

```bash
python src/lidar_stability/pipeline/build_enhanced_ground_truth.py --help
```

Funcion:
- Genera ground truth enriquecido desde archivos featured.
- Permite multiples corridas en una ejecucion con `--run-config` (hiperparametros por corrida).
- Por defecto guarda en `output/models` con salida compacta (bundle comprimido + metricas consolidadas).

## LiDAR

### Compute terrain features

Archivo: `src/lidar_stability/lidar/compute_route_terrain_features.py`

Uso:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py --help
```

Funcion:
- Extrae features de terreno por ruta map-matched.
- Incluye modo archivo y modo batch por DOBACK.

### Download CNIG tiles

Archivo: `src/lidar_stability/lidar/download_cnig_lidar_tiles.py`

Uso:

```bash
python src/lidar_stability/lidar/download_cnig_lidar_tiles.py --help
```

Funcion:
- Descarga y reporta tiles LiDAR CNIG pendientes.

## Visualizacion

### Visualize route on LiDAR (2D)

Archivo: `src/lidar_stability/visualization/visualize_route_lidar.py`

Uso:

```bash
python src/lidar_stability/visualization/visualize_route_lidar.py --help
```

### Visualize interactive 3D

Archivo: `src/lidar_stability/visualization/visualize_3d_interactive.py`

Uso:

```bash
python src/lidar_stability/visualization/visualize_3d_interactive.py --help
```

### Process raw data utility

Archivo: `src/lidar_stability/visualization/process_raw_data.py`

Uso:

```bash
python src/lidar_stability/visualization/process_raw_data.py --help
```

## ML

### Train baseline w model

Archivo: `src/lidar_stability/ml/train_w_model.py`

Uso:

```bash
python src/lidar_stability/ml/train_w_model.py --help
```

### Train multiple models

Archivo: `src/lidar_stability/ml/train_models_cli.py`

Uso:

```bash
python src/lidar_stability/ml/train_models_cli.py --help
```

### Plot leaderboard

Archivo: `src/lidar_stability/ml/plot_models_leaderboard.py`

Uso:

```bash
python src/lidar_stability/ml/plot_models_leaderboard.py --help
```

Funcion:
- Lee todos los `*_metrics.json` de la carpeta indicada.
- Genera tabla/plots y reporte HTML interactivo (`--html-file`, por defecto `leaderboard.html`).
- Por defecto usa modo compacto: guarda HTML + JSON (menos archivos). Con `--no-compact-output` tambien exporta CSV y PNG.

## Physics

### Compare stability CSV

Archivo: `src/lidar_stability/physics/compare_stability_csv.py`

Uso:

```bash
python src/lidar_stability/physics/compare_stability_csv.py --help
```

Funcion:
- Compara SI fisico vs SI medido con opciones de ajuste y diagnostico.

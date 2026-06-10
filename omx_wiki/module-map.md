---
title: Module Map
category: reference
tags: [modules, code-map, api]
---

# Module Map

## `config`

### `device_registry.py`

- `DeviceRegistry`: carga YAMLs `doback-*.yaml`.
- `get_config(device_id)`: devuelve configuracion por DOBACK.
- `get_device_from_filename(filename)`: extrae `23`, `24`, `27`, `28` desde nombres de archivo.
- `validate_constants(device_id)`: comprueba campos requeridos.
- `get_registry()`: singleton lazy.

### `device_constants.py`

- `resolve_device_config_from_filename`: resuelve config desde nombre de CSV.
- `extract_literal_device_constants_from_filename`: extrae constantes literales.
- `assign_literal_device_constants`: anade columnas de constantes a un dataframe.

## `parsers`

### `batch_processor.py`

Nucleo de ingestion y limpieza. Produce `processed-data`.

Funciones clave:

- `parse_gps_file`
- `parse_stability_file`
- `filter_isolated_points`
- `detect_imu_outliers_rolling`
- `match_by_timestamp`
- `split_into_segments`
- `process_all`

### `map_matching.py`

Map-matching completo contra red viaria.

Funciones clave:

- `load_network_from_graphml`
- `load_network_from_osmnx`
- `get_network_for_bbox`
- `build_spatial_index`
- `match_track`
- `process_files`

### `route_visualizer.py`

Mapa HTML sencillo de rutas coloreadas por SI.

## `lidar`

### `laz_reader.py`

- `LAZReader`: carga `.laz`, obtiene bounds, stats, patches por radio y KNN.

### `tif_reader.py`

- `TIFReader`: lee raster TIF, consulta elevacion y extrae patches.

### `terrain_provider.py`

- `TerrainProvider`: fachada que combina LAZ y TIF.

### `terrain_features.py`

- `TerrainFeatureExtractor`: calcula features sobre DEM local.

### `compute_route_terrain_features.py`

CLI y funciones de enriquecimiento:

- `find_relevant_laz_tiles`
- `extract_terrain_features_at_point`
- `enrich_route_with_terrain_features`
- `enrich_doback_batch`

## `physics`

### `stability_engine.py`

- `StabilityEngine`: calcula angulo critico y SI estatico.

Metodos:

- `critical_angle`
- `si_static`
- `si_static_batch`
- `si_static_from_deg`
- `get_vehicle_params`

## `pipeline`

### `ground_truth.py`

- `build_ground_truth`: ground truth basico con `si_real`, `si_static`, `delta_si`.
- `build_enhanced_ground_truth`: ground truth enriquecido con SI IMU/LiDAR/fused y dinamica observada.
- `export_ground_truth`: export CSV.

### `audit_pipeline_coverage.py`

Audita presencia de rutas en raw, processed, map-matched y featured.

## `ekf`

- `ExtendedKalmanFilter`: estado `[x, y, speed, heading]`.
- `calculate_imu_absolute_timestamp`: convierte `t_us` a timestamps absolutos.
- `merge_gps_imu`: merge asof GPS/IMU.
- `match_gps_stability`: alinea GPS sobre timeline de estabilidad.
- `split_segments`: segmentacion espacial.

## `ml`

- `feature_engineering.py`: construye `X`, `y`, features usadas y dataframe limpio.
- `train_w_model.py`: baseline RF.
- `train_models_cli.py`: entrenamiento multi-modelo.
- `adaptive_hyperparam_search.py`: Optuna, resume y export de historial.
- `plot_models_leaderboard.py`: carga metricas/historiales en dataframe.

## `visualization`

- `segment_ranking.py`: ranking de segmentos.
- `segment_loader.py`: carga featured + busca LAZ.
- `point_cloud_processor.py`: filtra y decima nube de puntos.
- `map_builder.py`: crea HTML/PNG 2D/3D.
- `cli.py`: flujo interactivo.
- `visualize_segment.py`: flujo directo por segmento.
- `compare_models_metrics.py`: graficas de metricas de modelos.


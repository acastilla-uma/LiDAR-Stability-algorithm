# Project Status

Last update: 2026-03-20

## Resumen ejecutivo

- Arquitectura consolidada bajo `src/lidar_stability`.
- Pipeline principal operativo con rutas:
  - `Doback-Data/processed-data`
  - `Doback-Data/map-matched`
  - `Doback-Data/featured`
- CLI principales validadas con `--help` tras reorganizacion.
- Auditoria de cobertura de pipeline corregida para resolver base del repo correctamente.
- Extraccion de features de terreno optimizada en hotspots computacionales.

## Estado por modulo

1. Parsers
- `batch_processor.py`: operativo, rutas por defecto actualizadas a `processed-data`.
- `map_matching.py`: operativo, base del repo corregida.
- `route_visualizer.py`: operativo, ejemplos y rutas actualizadas.
- `map_matcher.py`: cache de red movida a `output/road_network.graphml` con base correcta.

2. LiDAR
- `compute_route_terrain_features.py`: operativo, optimizado y con resolucion de rutas corregida.
- `terrain_features.py`: optimizado en `phi_lidar` y `ruggedness`.
- `laz_reader.py`, `tif_reader.py`, `terrain_provider.py`: integrados y en uso por pipeline.

3. Pipeline
- `run_full_pipeline.py`: rutas base corregidas para layout actual.
- `audit_pipeline_coverage.py`: salida simple corregida y diagnostico cuando no hay pares raw.
- `build_enhanced_ground_truth.py`: imports y base de repo corregidos.

4. ML
- `train_w_model.py`, `train_models_cli.py`, `plot_models_leaderboard.py`: imports y resolucion de repo corregidos.
- Feature engineering y entrenamiento listos para ejecucion por CLI.

5. Physics
- `compare_stability_csv.py`: rutas/imports corregidos y config por defecto apuntando a `src/lidar_stability/config/vehicle.yaml`.

6. Visualizacion
- `visualize_route_lidar.py`, `visualize_3d_interactive.py`, `run_examples.py`: rutas/imports corregidos.

## Validaciones recientes

- Validacion funcional de map-matching sobre archivo real:
  - Entrada: `Doback-Data/processed-data/DOBACK023_20251012_seg1.csv`
  - Salida: `output/smoke/map-matched/DOBACK023_20251012_seg1.csv`
  - Resultado: OK.
- Verificacion de CLIs (arranque por ayuda) para modulos principales: OK.
- Comprobacion de errores en archivos modificados: sin errores de analisis en IDE.

## Riesgos y pendientes

- La extraccion de features LiDAR puede ser pesada en runtime en escenarios de alta densidad.
- Conviene mantener perfiles de ejecucion "smoke" (sampling alto, dem pequeño) para validaciones rapidas.
- Recomendable añadir benchmarks automatizados y pruebas de regresion de rendimiento para terrain features.

## Proximos pasos recomendados

1. Ejecutar smoke end-to-end sobre 1 ruta por dispositivo DOBACK.
2. Consolidar metrica de tiempo por etapa (processed/map-matched/featured).
3. Publicar comparativa de rendimiento antes/despues en `output/results`.
4. Ampliar pruebas para confirmar consistencia de salidas en reorganizacion completa.

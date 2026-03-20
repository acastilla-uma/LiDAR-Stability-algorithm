# LiDAR Stability Algorithm

Pipeline para procesar rutas DOBACK con datos GPS/estabilidad, hacer map-matching, enriquecer con terreno LiDAR y generar salidas de visualizacion y modelado.

## Estado actual

- Estructura de codigo consolidada en `src/lidar_stability`.
- Flujo principal operativo: `processed-data -> map-matched -> featured`.
- Scripts CLI actualizados para resolver rutas del repo correctamente tras la reorganizacion.
- Extraccion de features de terreno optimizada para reducir tiempo de ejecucion.

## Instalacion

```bash
pip install -r requirements.txt
```

## Pipeline completo recomendado

```bash
python src/lidar_stability/pipeline/run_full_pipeline.py \
  --base DOBACK024_20250929 \
  --data-dir Doback-Data \
  --processed-dir Doback-Data/processed-data \
  --mapmatched-dir Doback-Data/map-matched \
  --featured-dir Doback-Data/featured \
  --output-dir output \
  --laz-dir LiDAR-Maps/cnig \
  --points-sample 700000
```

Salidas esperadas:

- `Doback-Data/processed-data/*.csv`
- `Doback-Data/map-matched/*.csv`
- `Doback-Data/featured/*.csv`
- `output/<BASE>_final_2d.png`
- `output/<BASE>_final_3d.html`

## Flujo por etapas

1. Procesado batch de crudos:

```bash
python src/lidar_stability/parsers/batch_processor.py \
  --data-dir Doback-Data \
  --output-dir Doback-Data/processed-data
```

2. Map-matching:

```bash
python src/lidar_stability/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched
```

3. Enriquecimiento de features de terreno:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --mapmatch Doback-Data/map-matched/DOBACK024_20250929_seg11.csv \
  --laz-dir LiDAR-Maps/cnig \
  --output Doback-Data/featured/DOBACK024_20250929_seg11.csv
```

4. Visualizacion 2D:

```bash
python src/lidar_stability/visualization/visualize_route_lidar.py \
  --mapmatch Doback-Data/featured/DOBACK024_20250929_seg11.csv \
  --output output/ruta_seg11_2d.png
```

5. Visualizacion 3D:

```bash
python src/lidar_stability/visualization/visualize_3d_interactive.py \
  --base DOBACK024_20250929 \
  --points-sample 700000 \
  --output output/ruta_3d_DOBACK024_20250929.html
```

## Documentacion

- `docs/QUICK_START.md`
- `docs/PROJECT_STATUS.md`
- `docs/ROADMAP.md`
- `docs/guides/map_matching.md`
- `docs/guides/TERRAIN_FEATURES_EXTRACTION.md`
- `docs/guides/TRAINING_EVALUATION_GUIDE.md`
- `docs/guides/CLI_REFERENCE.md`

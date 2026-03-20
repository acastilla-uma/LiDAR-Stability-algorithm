# Quick Start

## 1) Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2) Comando unico recomendado

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

Genera automaticamente:

- `Doback-Data/processed-data/*.csv`
- `Doback-Data/map-matched/*.csv`
- `Doback-Data/featured/*.csv`
- `output/<BASE>_final_2d.png`
- `output/<BASE>_final_3d.html`

## 3) Pipeline por etapas

```bash
python src/lidar_stability/parsers/batch_processor.py --data-dir Doback-Data --output-dir Doback-Data/processed-data
python src/lidar_stability/parsers/map_matching.py --input Doback-Data/processed-data --output Doback-Data/map-matched
python src/lidar_stability/lidar/compute_route_terrain_features.py --mapmatch Doback-Data/map-matched/DOBACK024_20250929_seg11.csv --laz-dir LiDAR-Maps/cnig --output Doback-Data/featured/DOBACK024_20250929_seg11.csv
python src/lidar_stability/visualization/visualize_route_lidar.py --mapmatch Doback-Data/featured/DOBACK024_20250929_seg11.csv --output output/ruta_seg11_2d.png
python src/lidar_stability/visualization/visualize_3d_interactive.py --base DOBACK024_20250929 --points-sample 700000 --output output/ruta_3d_DOBACK024_20250929.html
```

## 4) Verificacion rapida de CLIs

```bash
python src/lidar_stability/pipeline/run_full_pipeline.py --help
python src/lidar_stability/parsers/map_matching.py --help
python src/lidar_stability/lidar/compute_route_terrain_features.py --help
```

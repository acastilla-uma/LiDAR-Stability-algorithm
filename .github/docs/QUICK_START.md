# Quick Start

## Comando único recomendado

```bash
python Scripts/pipeline/run_full_pipeline.py \
  --base DOBACK024_20250929 \
  --data-dir Doback-Data \
  --processed-dir Doback-Data/processed-data \
  --mapmatched-dir Doback-Data/map-matched \
  --output-dir output \
  --points-sample 700000
```

Genera automáticamente:
- `Doback-Data/processed-data/*.csv`
- `Doback-Data/map-matched/*.csv`
- `output/<BASE>_final_2d.png`
- `output/<BASE>_final_3d.html`

## Verificación rápida

```bash
python Scripts/tests/run_visual_tests.py
```

## Alternativa por pasos

```bash
python Scripts/parsers/batch_processor.py --data-dir Doback-Data --output-dir Doback-Data/processed-data
python Scripts/parsers/map_matching.py --input Doback-Data/processed-data --output Doback-Data/map-matched
python Scripts/visualization/visualize_route_lidar.py --mapmatch Doback-Data/map-matched/DOBACK024_20250929_seg11.csv --output output/ruta_seg11_2d.png
python Scripts/visualization/visualize_3d_interactive.py --base DOBACK024_20250929 --points-sample 700000 --output output/ruta_3d_DOBACK024_20250929.html
```

# LiDAR Stability Algorithm

Pipeline práctico para procesar rutas y generar visualización final 2D + 3D.

## Requisitos

```bash
pip install -r requirements.txt
```

## Comando único (raw → processed-data → map-matched → featured → 2D + 3D)

```bash
python Scripts/pipeline/run_full_pipeline.py \
  --base DOBACK024_20250929 \
  --data-dir Doback-Data \
  --processed-dir Doback-Data/processed-data \
  --mapmatched-dir Doback-Data/map-matched \
  --featured-dir Doback-Data/featured \
  --output-dir output \
  --points-sample 700000
```

Salida final:
- `Doback-Data/processed-data/*.csv`
- `Doback-Data/map-matched/*.csv`
- `Doback-Data/featured/*.csv`
- `output/<BASE>_final_2d.png`
- `output/<BASE>_final_3d.html`

## Pasos manuales (opcional)

### 1) Raw → processed-data

```bash
python Scripts/parsers/batch_processor.py \
  --data-dir Doback-Data \
  --output-dir Doback-Data/processed-data
```

### 2) processed-data → map-matched

```bash
python Scripts/parsers/map_matching.py \
  --input Doback-Data/processed-data \
  --output Doback-Data/map-matched
```

### 3) Visualización 2D

```bash
python Scripts/visualization/visualize_route_lidar.py \
  --mapmatch Doback-Data/featured/DOBACK024_20250929_seg11.csv \
  --output output/ruta_seg11_2d.png
```

### 4) Visualización 3D interactiva

```bash
python Scripts/visualization/visualize_3d_interactive.py \
  --base DOBACK024_20250929 \
  --points-sample 700000 \
  --output output/ruta_3d_DOBACK024_20250929.html
```

Controles 3D:
- Rotar: click + arrastrar
- Zoom: rueda del ratón
- Pan: Shift + click + arrastrar

### 5) Descargar teselas CNIG LiDAR 3ª cobertura

Usando la lista generada en `output/cnig_missing_tiles_all_doback.txt`:

```bash
python Scripts/lidar/download_cnig_lidar_tiles.py \
  --list output/cnig_missing_tiles_all_doback.txt \
  --dest LiDAR-Maps/cnig
```

Prueba en seco (sin descargar):

```bash
python Scripts/lidar/download_cnig_lidar_tiles.py \
  --list output/cnig_missing_tiles_all_doback.txt \
  --dry-run \
  --max-files 10
```

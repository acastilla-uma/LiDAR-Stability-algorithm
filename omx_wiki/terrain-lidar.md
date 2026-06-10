---
title: Terrain And LiDAR
category: architecture
tags: [lidar, terrain, features, laz, tif]
---

# Terrain And LiDAR

El enriquecimiento LiDAR convierte una ruta map-matched en un dataset `featured` con medidas locales de terreno.

## Archivos clave

- `src/lidar_stability/lidar/laz_reader.py`
- `src/lidar_stability/lidar/tif_reader.py`
- `src/lidar_stability/lidar/terrain_provider.py`
- `src/lidar_stability/lidar/terrain_features.py`
- `src/lidar_stability/lidar/compute_route_terrain_features.py`

## Flujo de `compute_route_terrain_features.py`

1. Lee CSV map-matched.
2. Para cada punto, localiza tiles LAZ cercanos con `find_relevant_laz_tiles`.
3. Usa `LAZReader.extract_patch` para extraer puntos dentro de `search_radius`.
4. Une puntos de varios tiles si hace falta.
5. Interpola elevacion sobre un DEM local de `dem_size x dem_size`.
6. Calcula features de terreno con `TerrainFeatureExtractor`.
7. Anade constantes literales DOBACK.
8. Escribe CSV featured.

## Features generadas

- `phi_lidar`: pendiente transversal estimada.
- `phi_lidar_deg`: pendiente transversal en grados.
- `tri`: Terrain Roughness Index.
- `ruggedness`: metrica de rugosidad/variacion local.
- `z_min`, `z_max`, `z_mean`, `z_std`, `z_range`: estadisticos de elevacion.
- `n_points_used`: puntos LiDAR usados en el patch.

## Parametros importantes

- `--search-radius`: radio en metros para buscar puntos LiDAR alrededor de cada punto de ruta.
- `--dem-size`: resolucion del DEM local.
- `--vehicle-track`: ancho de via usado para pendiente transversal.
- `--sampling`: procesa uno de cada N puntos.

## Rendimiento

La etapa LiDAR suele ser la mas pesada. Para pruebas rapidas:

- bajar `--dem-size`
- subir `--sampling`
- usar una ruta corta
- limitar a un solo segmento

Ejemplo smoke:

```bash
python src/lidar_stability/lidar/compute_route_terrain_features.py \
  --mapmatch Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
  --laz-dir LiDAR-Maps/cnig \
  --output output/smoke_featured.csv \
  --search-radius 50 \
  --dem-size 64 \
  --sampling 10
```

## Fallos esperables

- Sin tiles cerca: las features quedan en `NaN`.
- Pocos puntos: `n_points_used` bajo y features no fiables.
- Interpolacion fallida: se devuelve un bloque de features con `NaN`.
- Tiles corruptos: se registran warnings y se continua con otros tiles.


---
title: Architecture
category: architecture
tags: [architecture, modules, flow]
---

# Architecture

La arquitectura esta organizada como un pipeline de datos con capas bastante claras.

```text
raw files
  |
  v
parsers
  |
  v
map-matched CSVs
  |
  v
lidar terrain enrichment
  |
  v
featured CSVs
  |
  +--> pipeline/ground_truth
  +--> ml training
  +--> visualization
```

## Capas

### 1. Ingestion y limpieza

Paquete: `src/lidar_stability/parsers`

Responsabilidades:

- Parsear ficheros GPS y estabilidad.
- Validar rangos basicos.
- Detectar outliers.
- Registrar auditoria de filas descartadas.
- Unir GPS y estabilidad por timestamp.
- Dividir rutas en segmentos.
- Hacer map-matching.

Pagina relacionada: [[data-pipeline]].

### 2. LiDAR y terreno

Paquete: `src/lidar_stability/lidar`

Responsabilidades:

- Leer LAZ y TIF.
- Buscar tiles LiDAR relevantes para cada punto.
- Extraer patches alrededor de la ruta.
- Interpolar DEM local.
- Calcular features de terreno.

Pagina relacionada: [[terrain-lidar]].

### 3. Configuracion

Paquete: `src/lidar_stability/config`

Responsabilidades:

- Resolver configuracion por dispositivo DOBACK.
- Extraer constantes literales desde nombres de fichero.
- Cargar YAMLs de `config/devices`.

Pagina relacionada: [[configuration-and-data]].

### 4. Fisica y ground truth

Paquetes:

- `src/lidar_stability/physics`
- `src/lidar_stability/pipeline`

Responsabilidades:

- Calcular angulo critico.
- Calcular SI estatico.
- Construir ground truth basico y enriquecido.
- Auditar cobertura de pipeline.

### 5. ML

Paquete: `src/lidar_stability/ml`

Responsabilidades:

- Construir datasets de entrenamiento.
- Entrenar RandomForest, ExtraTrees y GradientBoosting.
- Ejecutar busqueda adaptativa con Optuna.
- Leer metricas para leaderboard.

Pagina relacionada: [[ml-training]].

### 6. Visualizacion

Paquete: `src/lidar_stability/visualization`

Responsabilidades:

- Ranking de segmentos por impacto/error.
- Carga de segmento featured + LiDAR.
- Procesamiento y decimacion de nube de puntos.
- Construccion de mapas 2D/3D.

Pagina relacionada: [[visualization]].

## Convenciones de dependencias

- Los CLIs suelen resolver la raiz del repo a partir de `Path(__file__)`.
- Las salidas generadas deben ir a `output/` o a carpetas de datos, no a `src/`.
- Los datos pesados estan ignorados por git.
- Las pruebas viven en `tests/` y cubren sprints historicos del proyecto.


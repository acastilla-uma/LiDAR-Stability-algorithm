---
title: Glossary
category: reference
tags: [glossary, terms]
---

# Glossary

## DOBACK

Familia de dispositivos/vehiculos del dataset. El repo reconoce al menos `DOBACK023`, `DOBACK024`, `DOBACK027` y `DOBACK028`.

## SI

Stability Index. Indice de estabilidad usado para colorear rutas, calcular ground truth y evaluar riesgo.

## `si_real`

SI observado o medido, normalmente derivado de `si` o `si_mcu`.

## `si_static`

SI calculado desde el modelo fisico estatico usando roll y angulo critico.

## `delta_si`

Diferencia entre SI observado y SI estatico.

## `gy`

Variable target por defecto en el pipeline ML. Se interpreta como giro/omega observado para el modelo dinamico.

## Featured

CSV final enriquecido con datos de ruta, estabilidad, terreno LiDAR y constantes del dispositivo.

## Map-matching

Proceso de proyectar puntos GPS sobre una red viaria para obtener coordenadas y asignaciones mas coherentes.

## LAZ

Formato comprimido de nube de puntos LiDAR.

## TIF

Raster geoespacial, usado como fuente alternativa o complementaria de elevacion.

## `phi_lidar`

Pendiente transversal estimada desde el terreno LiDAR.

## TRI

Terrain Roughness Index. Metrica de rugosidad del terreno.

## Ruggedness

Metrica de variabilidad local del terreno.

## Holdout

Particion final no usada para entrenar ni optimizar CV; sirve para medir generalizacion.

## Generalization gap

Diferencia entre metrica CV y metrica holdout. Gap alto sugiere sobreajuste o leakage.


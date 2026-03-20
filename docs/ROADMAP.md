# Roadmap Operativo

Ultima actualizacion: 2026-03-20

## Objetivo

Consolidar el pipeline PIML con estructura `src/`, asegurar trazabilidad por etapa y estabilizar rendimiento en extraccion de features de terreno.

## Horizonte corto (1-2 semanas)

- [x] Corregir resolucion de rutas del repo tras reorganizacion a `src/lidar_stability`.
- [x] Corregir imports de scripts CLI para ejecutar por ruta relativa del repositorio.
- [x] Corregir auditoria de cobertura para no quedar en salida vacia.
- [x] Estandarizar carpeta `Doback-Data/processed-data` en scripts y docs.
- [x] Optimizar hotspots de terrain features (`phi_lidar`, `ruggedness`, escritura en DataFrame).
- [ ] Ejecutar smoke E2E automatizado para 1 archivo por dispositivo DOBACK.
- [ ] Publicar informe de tiempos por etapa en `output/results`.

## Horizonte medio (3-6 semanas)

- [ ] Añadir pruebas de regresion de rendimiento para terrain feature extraction.
- [ ] Definir perfiles de ejecucion `smoke`, `standard` y `full` para CLI pesadas.
- [ ] Integrar pipeline de entrenamiento con entradas featured ya auditadas.
- [ ] Estandarizar reportes de metricas ML y leaderboard en una salida comun.
- [ ] Añadir validacion de cobertura de datos por DOBACK en CI local.

## Horizonte largo (6+ semanas)

- [ ] Integrar mapas de transitabilidad con evaluacion por yaw multiorientacion.
- [ ] Preparar modo batch de inferencia para grandes superficies raster.
- [ ] Publicar paquete reproducible de entrenamiento + inferencia para transferencia.

## Definition of Done por tarea

Una tarea se considera cerrada solo cuando:

1. El script/flujo ejecuta sin errores en su comando principal.
2. Las rutas de entrada/salida son validas en estructura actual del repositorio.
3. La documentacion de uso queda actualizada en `docs/`.
4. El cambio queda registrado en `CHANGELOG.md`.

## Checklist de control de reorganizacion

- [x] CLI de pipeline principal
- [x] CLI de parsers
- [x] CLI de LiDAR
- [x] CLI de ML
- [x] CLI de visualizacion
- [x] CLI de fisica
- [x] Documentacion base (README/QUICK_START/STATUS/ROADMAP)
- [x] Guías tecnicas existentes
- [x] Changelog actualizado

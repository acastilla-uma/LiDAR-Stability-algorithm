# 📋 PLAN_ORIGINAL.md - Versión Íntegra del Plan Inicial

Este archivo preserva el **plan inicial íntegro** del proyecto antes de su transformación en el roadmap de Sprints.

---

## Contexto General

**Fecha:** 23 de febrero de 2026  
**Proyecto:** LiDAR-Stability-Algorithm - PIML Traversability Mapping  
**Rol ejecutivo:** Technical Project Manager + Senior AI/Robotics Engineer

---

## El Proyecto y Entregable Final

**Objetivo:** Evaluar el riesgo de vuelco de vehículos pesados en terrenos off-road.

**Entregable final:** Un script que procese un mapa topográfico masivo y genere un **Mapa 2D de Transitabilidad** (Traversability Map). En este raster de salida, cada píxel representará el **Índice de Estabilidad** ($SI$) predicho, sirviendo como mapa de costes para navegación autónoma (RL).

---

## Metodología Estricta: Physics-Informed Machine Learning (PIML)

El proyecto usará **una arquitectura híbrida donde la IA no sustituye a la física, sino que la complementa**:

### Módulo Físico (Determinista)
Calcula la cinemática, la inclinación estática del chasis y la estabilidad base.

### Módulo ML (Predictivo)
Solo predice la "perturbación dinámica" ($\Delta SI$) causada por la velocidad y la rugosidad rebotando en la suspensión.

### Ecuación Final
$$SI_{final} = SI_{estatico\_calculado} + \Delta SI_{dinamico\_predicho}$$

---

## Detalles de Implementación por Módulos (Requisitos de Arquitectura)

### Módulo 1: Fusión de Sensores (EKF)

**Objetivo:** Sincronizar GPS (1Hz) e IMU (50Hz+).

**Lógica:** Implementar un Filtro de Kalman Extendido. El vector de estado estimará $(x, y, v, \psi)$. 
- **Fase de Predicción:** Usar las ecuaciones cinemáticas con los datos de alta frecuencia de la IMU ($a_x, a_y, \dot{\psi}$) para estimar la posición. 
- **Fase de Actualización:** Usar el GPS a 1Hz para corregir la deriva.

### Módulo 2: Motor Físico (Ground Truth & Estabilidad Estática)

**Lógica:** A partir de los parámetros del vehículo (Masa $M$, Altura CG $H_g$, Ancho vía $S$), calcular el ángulo crítico de vuelco:
$$\phi_c = \arctan\left(\frac{S}{2 \cdot H_g}\right)$$

**GroundTruth:** Usar los datos de la IMU ($a_y$, ángulos Euler) para calcular el $SI_{real}$ dividiendo los momentos de vuelco entre los momentos estabilizadores. Si $SI_{real} \ge 1.0$, es vuelco.

### Módulo 3: Procesamiento LiDAR (Extracción de Geometría)

**Lógica:** Para cada coordenada UTM de la trayectoria y su Yaw ($\psi$), extraer un parche del mapa (Raster .tif o Nube .laz filtrada).

**Cálculos:** Extraer la inclinación transversal topográfica ($\phi_{lidar}$) y el **Índice de Rugosidad del Terreno**:
$$TRI = \sqrt{\frac{\sum (Z_{ij} - \bar{Z})^2}{N}}$$

### Módulo 4: Machine Learning y Sim-to-Real

**Lógica:** Se usará un simulador (Gazebo/Project Chrono) validado con los datos reales seguros para generar datos sintéticos de vuelco y balancear el dataset (Data Augmentation).

**Target del ML:** El modelo (XGBoost/RF) entrenará para predecir el residuo dinámico: 
$$\text{Target} = SI_{real} - SI_{estatico}$$

**Inputs del ML:** $[\text{Velocidad}, TRI, \text{Roll}_{LiDAR}, \text{Pitch}_{LiDAR}, \text{Masa}, \text{Tipo\_Suspension}]$

### Módulo 5: Generador del Mapa 2D

**Lógica:** Bucle vectorizado (ej. numpy, rasterio) aplicando una "sliding window" sobre el DTM. 

Por cada píxel, evalúa **8 direcciones de Yaw** a una velocidad constante. Calcula el $SI_{final}$ máximo (Física + ML) y colorea el GeoTIFF de salida:
- Rojo si $SI > 0.85$
- Verde si $SI < 0.4$

---

## Estructura del Repositorio Inicial

```
/
├── scripts/                    # Código Python (EKF, PIML, Generador de Mapas)
├── mapas_lidar/
│   ├── geoportal_madrid/       # Archivos Raster DTM (.tif)
│   └── cnig/                   # Nubes de puntos crudas (.laz)
└── datos_doback/
    ├── estabilidad/            # Archivos TXT de IMU/Estabilidad (>50Hz)
    └── gps/                    # Archivos TXT de localización (1Hz)
```

---

## Instrucción Operativa

Utilizando todo este contexto técnico, las fórmulas y la arquitectura PIML, se generaría el Roadmap del proyecto estructurado en los siguientes **Sprints lógicos**:

### Sprint 1
Setup, Motor Físico Base y Ground Truth ($SI_{real}$)

### Sprint 2
Fusión de Sensores (EKF para GPS+IMU)

### Sprint 3
Procesamiento LiDAR unificado (TIF/LAZ, $\phi_{lidar}$ y $TRI$)

### Sprint 4
Sim-to-Real (Data Augmentation de vuelcos en simulador)

### Sprint 5
Módulo ML predictivo ($\Delta SI$)

### Sprint 6
Generador masivo de Mapas 2D de Transitabilidad (Sliding window GeoTIFF)

**Cada tarea debe tener tests ejecutables por CLI para cumplir con el Definition of Done.**

---

## Formato de Salida Requerido

El roadmap debe cumplir con:

1. **Conservar el plan original** mediante este archivo PLAN_ORIGINAL.md
2. **Descomponer en Sprints con tareas accionables** usando casillas `- [ ]`
3. **Criterios de Aceptación (Tests):** Comandos concretos y ejecutables
4. **Definition of Done (DoD):**
   - *Nivel Tarea:* Una tarea solo se marca como completa si sus tests de verificación pasan al 100%
   - *Nivel Sprint:* Un Sprint solo se archiva como completado tras verificar con éxito el 100% de sus tareas

---

## Decisiones Técnicas Tomadas

| Decisión | Valor |
|----------|-------|
| **Ground Truth SI** | Columna `si` del MCU directamente (no recalculado) |
| **Simulador** | Project Chrono |
| **Entorno Python** | pip + requirements.txt (sin virtualenv) |
| **Vehículo (DOBACK024)** | M=18000 kg, S=2.480 m, Hg=1.850 m, Ixx=89300 kg·m² |
| **Frecuencia real IMU** | 10 Hz (confirmado por análisis de datos) |
| **CRS Geoespacial** | ETRS89 / UTM Zone 30N (EPSG:25830) |

---

## Hallazgos de Investigación del Workspace

### Estado Actual del Proyecto
- **Madurez:** Empty scaffold — cero código Python, directorio Scripts/ vacío
- **Dependencias:** Sin requirements.txt, setup.py, o configuración

### Datos Disponibles

**GPS (Doback-Data/GPS/):**
- 2 archivos: GPS_DOBACK027_20250814_0.txt (1398 líneas), GPS_DOBACK027_20250814_1.txt (395 líneas)
- Formato: CSV con separador coma
- Frecuencia: ~1 Hz (como especificado)
- Problemas: Abundantes filas sin fix, corrupción frecuente (truncación, valores absurdos)

**IMU/Estabilidad (Doback-Data/Stability/):**
- 2 archivos: ESTABILIDAD_DOBACK024_20250825_188.txt (22773 líneas), ESTABILIDAD_DOBACK024_20250825_189.txt (20927 líneas)
- Formato: Campos separados por `;`
- Frecuencia: ~10 Hz (NO 50 Hz como inicialmente asumido)
- Estructura: 19 columnas por fila, incluyendo columna `si` del MCU

**LiDAR (LiDAR-Maps/):**
- CNIG/: 51 archivos .laz PNOA 2024 (ETRS89 UTM 30N)
- geo-mad/: 1 raster .tif (447-4483.tif)

### Nota Importante
**GPS y Estabilidad provienen de dispositivos distintos (DOBACK027 vs DOBACK024) y fechas distintas.** La arquitectura debe manejar datasets independientes.

---

## Referencias y Normativas

- **Physics:** Rollover moment ratio (SI = tan(roll) / tan(φc))
- **LiDAR Data Source:** IGN PNOA 2024 (ETRS89 UTM Zone 30N)
- **Sensor Configuration:** MCU with high-precision IMU
- **ML Framework:** XGBoost + Random Forest
- **Simulation:** Project Chrono (PyChrono) for vehicle dynamics

---

**Documento generado:** 23 de febrero de 2026  
**Versión del Plan:** 1.0 - Plan inicial íntegro  
**Transformación en Roadmap:** ✅ Completada en ROADMAP.md

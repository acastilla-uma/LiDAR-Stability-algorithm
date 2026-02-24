# 🗺️ Roadmap del Proyecto - LiDAR Stability Algorithm PIML

> **Definition of Done (DoD):** Toda tarea requiere un 100% de éxito en sus tests para cerrarse. Un Sprint se archiva únicamente al verificar el 100% de sus tareas.

---

## 📋 Resumen Ejecutivo

**Objetivo:** Evaluar el riesgo de vuelco de vehículos pesados en terrenos off-road mediante un mapa 2D de transitabilidad (GeoTIFF) donde cada píxel representa el Índice de Estabilidad ($SI$) predicho.

**Metodología:** Physics-Informed Machine Learning (PIML) - arquitectura híbrida donde:
$$SI_{final} = SI_{estatico\_calculado} + \Delta SI_{dinamico\_predicho}$$

**Parámetros del Vehículo (DOBACK024):**
- Masa: 18000 kg
- Ancho de vía: 2.480 m
- Altura CG: 1.850 m
- Ángulo crítico: φc ≈ 33.8°

**Sensores:**
- GPS: 1 Hz (localización)
- IMU: 10 Hz (aceleración, rotación, estabilidad)
- LiDAR: Nubes de puntos CNIG PNOA 2024 + rasters DTM

---

## Sprint 1: Setup, Motor Físico Base y Ground Truth ($SI_{real}$) ✅ COMPLETADO

**Objetivo:** Establecer la infraestructura del proyecto, implementar el motor físico determinista y generar el dataset de ground truth etiquetado.

- [x] **Tarea 1.1: Scaffolding del proyecto**
  - *Descripción:* Crear estructura de directorios, requirements.txt, configuración del vehículo y control de versiones
  - *Subtareas:*
    - [x] Reorganizar árbol: `scripts/{config,parsers,physics,pipeline,ekf,lidar,ml,mapping,simulation,tests}`
    - [x] Crear `requirements.txt` (numpy, scipy, pandas, scikit-learn, xgboost, laspy, rasterio, pyproj, pytest, matplotlib, pyyaml)
    - [x] Crear `scripts/config/vehicle.yaml` con parámetros del Doback024
    - [x] Crear `.gitignore` (excluir .laz, .tif, datos masivos)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `python -c "import yaml; cfg=yaml.safe_load(open('scripts/config/vehicle.yaml')); assert cfg['vehicle']['mass_kg']==18000 and cfg['vehicle']['track_width_m']==2.480"`
    - [x] `pip install -r requirements.txt` completa sin errores
    - [x] Todas las librerías importan correctamente

- [x] **Tarea 1.2: Parser GPS robusto**
  - *Descripción:* Parsear archivos GPS_DOBACK*.txt, manejar filas sin fix, validar coordenadas y descartar datos corruptos
  - *Subtareas:*
    - [x] Crear `scripts/parsers/gps_parser.py` con función `parse_gps(filepath) → pd.DataFrame`
    - [x] Parsear metadata (device_id, session) de línea 1
    - [x] Manejar dos variantes de filas: con fix y sin fix
    - [x] Filtro de sanidad: lat/lon en rango peninsular, velocidad < 200 km/h, HDOP < 10
    - [x] Convertir fecha+hora GPS a datetime UTC
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint1.py::TestGPSParser::test_gps_parse_valid` ✅ PASSED
      - DataFrame no vacío, ~1398 registros válidos
      - Columnas correctas, sin NaN en lat/lon
    - [x] `pytest scripts/tests/test_sprint1.py::TestGPSParser::test_gps_rejects_corrupt` ✅ PASSED
      - Filas con velocidad 260807.90 km/h, lat truncada a 0.535, lon absurda que descartadas
    - [x] `pytest scripts/tests/test_sprint1.py::TestGPSParser::test_gps_metadata` ✅ PASSED
      - device_id == "DOBACK027", session == 0 extraídos correctamente

- [x] **Tarea 1.3: Parser IMU/Estabilidad robusto**
  - *Descripción:* Parsear ESTABILIDAD_DOBACK*.txt (sep. `;`), saltar líneas de timestamp y reinicio del MCU, exportar DataFrame limpio
  - *Subtareas:*
    - [x] Crear `scripts/parsers/imu_parser.py` con función `parse_imu(filepath) → pd.DataFrame`
    - [x] Detectar y saltar líneas de timestamp (`HH:MM:SSAM/PM`) y metadata
    - [x] Parsear 19 columnas por fila; descartar filas malformadas
    - [x] Calcular frecuencia: ~10 Hz (confirmado: 100 ms entre filas = 10 Hz, no 50 Hz como asumido)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint1.py::TestIMUParser::test_imu_parse_valid` ✅ PASSED
      - DataFrame con 22773 filas, 14+ columnas, si_mcu ∈ [0, 2]
    - [x] `pytest scripts/tests/test_sprint1.py::TestIMUParser::test_imu_skips_timestamps` ✅ PASSED
      - Ninguna fila contiene "AM" o "PM" en valores numéricos
    - [x] `pytest scripts/tests/test_sprint1.py::TestIMUParser::test_imu_frequency` ✅ PASSED
      - Mediana dt = 100.07 ms (confirma 10 Hz, dentro de [80, 120] ms)

- [x] **Tarea 1.4: Motor Físico — Ángulo crítico y $SI_{estático}$**
  - *Descripción:* Implementar ecuaciones física determinista: $\phi_c = \arctan(S / (2 \cdot H_g))$ y $SI_{est} = \tan(\phi_{roll}) / \tan(\phi_c)$
  - *Subtareas:*
    - [x] Crear `scripts/physics/stability_engine.py` con clase `StabilityEngine(config_path)`
    - [x] Método `critical_angle() → float` retorna 33.8° / 0.5899 rad
    - [x] Método `si_static(roll_rad) → float` — ratio de tangentes
    - [x] Método `si_static_batch(roll_array)` — versión vectorizada con NumPy
    - [x] Cargar parámetros del vehículo desde vehicle.yaml
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint1.py::TestStabilityEngine::test_critical_angle` ✅ PASSED
      - |φc − 33.8°| < 0.1°, |φc_rad − 0.5899| < 0.001 rad
    - [x] `pytest scripts/tests/test_sprint1.py::TestStabilityEngine::test_si_zero_roll` ✅ PASSED
      - si_static(0.0) == 0.0
    - [x] `pytest scripts/tests/test_sprint1.py::TestStabilityEngine::test_si_critical_angle` ✅ PASSED
      - si_static(φc) ≈ 1.0 (dentro de [0.95, 1.05])
    - [x] `pytest scripts/tests/test_sprint1.py::TestStabilityEngine::test_si_beyond_critical` ✅ PASSED
      - si_static(φc + 0.1) > 1.0

- [x] **Tarea 1.5: Pipeline de Ground Truth**
  - *Descripción:* Construir pipeline que une IMU + motor físico para generar dataset etiquetado. Columna si_mcu como $SI_{real}$, roll IMU para $SI_{estático}$, target ML es $\Delta SI = SI_{real} - SI_{estático}$
  - *Subtareas:*
    - [x] Crear `scripts/pipeline/ground_truth.py`
    - [x] Función `build_ground_truth(imu_df, engine) → DataFrame` con columnas [t_us, roll_deg, pitch_deg, si_real, si_static, delta_si]
    - [x] Validar si_real ∈ [0, 2] y δ_si tiene distribución razonable
    - [x] Exportar a CSV para inspección
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint1.py::TestGroundTruth::test_ground_truth_columns` ✅ PASSED
      - DataFrame contiene las 6 columnas esperadas
    - [x] `pytest scripts/tests/test_sprint1.py::TestGroundTruth::test_ground_truth_no_nans` ✅ PASSED
      - Cero NaN en columnas clave
    - [x] `pytest scripts/tests/test_sprint1.py::TestGroundTruth::test_delta_si_distribution` ✅ PASSED
      - mean(Δ SI) < 0.3 y std(Δ SI) < 0.5
    - [x] `pytest scripts/tests/test_sprint1.py::TestGroundTruth::test_ground_truth_consistency` ✅ PASSED
      - SI_final = SI_static + ΔSI verifica correctamente

**Resultado Sprint 1:** ✅ **17/17 tests PASSED** — Infraestructura completa, parsers validados, motor físico funcionando

---

## Sprint 2: Fusión de Sensores (EKF para GPS+IMU) ✅ COMPLETADO

**Objetivo:** Implementar Filtro de Kalman Extendido que fusa GPS de baja frecuencia (1 Hz) con IMU de alta frecuencia (10 Hz) para producir trayectoria continua suavizada.

- [x] **Tarea 2.1: Modelo cinemático del EKF**
  - *Descripción:* Implementar EKF con vector de estado x=[x_utm, y_utm, v, ψ]. Predicción con IMU (10 Hz), actualización con GPS (1 Hz)
  - *Subtareas:*
    - [x] Crear `scripts/ekf/ekf_fusion.py` clase `EKF(state_dim=4, meas_dim_gps=3)`
    - [x] Método `predict(ax, ay, gyro_z, dt)`: modelo cinemático de bicicleta
    - [x] Método `update(lat_gps, lon_gps, speed_gps)`: corrección GPS con conversión UTM
    - [x] Jacobiano $F$ analítico de transición de estado
    - [x] Matrices $Q$ (ruido proceso) y $R$ (ruido medida, escalado por HDOP)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint2.py::TestEKFStationary::test_ekf_initialization` ✅ PASSED
      - Estado inicial en cero, covarianza P bien formada
    - [x] `pytest scripts/tests/test_sprint2.py::TestEKFStationary::test_ekf_stationary` ✅ PASSED
      - Vehículo parado: posición converge a GPS ±1-5 m
    - [x] `pytest scripts/tests/test_sprint2.py::TestEKFStationary::test_ekf_constant_velocity` ✅ PASSED
      - Trayectoria rectilínea: velocidad 9-11 m/s tras sincronización
    - [x] `pytest scripts/tests/test_sprint2.py::TestEKFStationary::test_ekf_jacobian_numerical` ✅ PASSED
      - Jacobiano analítico vs numérico: error relativo < 1e-4

- [x] **Tarea 2.2: Sincronización temporal GPS ↔ IMU**
  - *Descripción:* Alinear GPS (fecha+hora UTC) e IMU (timeantwifi µs monotónico) usando marcas de referencia e interpolación
  - *Subtareas:*
    - [x] Crear `scripts/ekf/time_sync.py`
    - [x] Convertir timeantwifi a timestamp absoluto usando hora inicio de sesión
    - [x] Generar timeline unificada: muestras IMU cada ~100 ms, GPS a 1 Hz
    - [x] Función `merge_gps_imu(gps_df, imu_df) → DataFrame` con columna source (imu/gps)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint2.py::TestTimeSync::test_time_sync_monotonic` ✅ PASSED
      - Timestamps resultantes estrictamente crecientes
    - [x] `pytest scripts/tests/test_sprint2.py::TestTimeSync::test_time_sync_gps_ratio` ✅ PASSED
      - Merging produce número razonable de muestras (min ≤ merged ≤ 2x max)
    - [x] `pytest scripts/tests/test_sprint2.py::TestTimeSync::test_merge_preserves_all` ✅ PASSED
      - No se pierden muestras IMU ni GPS en el merge

- [x] **Tarea 2.3: Pipeline EKF end-to-end y visualización**
  - *Descripción:* Ejecutar EKF sobre sesiones reales. Generar trayectoria fusionada en UTM. Export a CSV y plots (GPS crudo vs EKF suavizado)
  - *Subtareas:*
    - [x] Crear `scripts/ekf/run_ekf.py` (CLI ejecutable con argparse)
    - [x] Función `run_ekf_session(gps_file, imu_file, config) → trajectory_df`
    - [x] Export trajectory como CSV: [t, x_utm, y_utm, v, yaw, source]
    - [x] Conversión GPS → UTM con pyproj (EPSG:25830 UTM 30N)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [x] `pytest scripts/tests/test_sprint2.py::TestEKFPipeline::test_ekf_pipeline_runs` ✅ PASSED
      - Pipeline completo ejecuta sin excepciones

**Resultado Sprint 2:** ✅ **8/8 tests PASSED** — EKF funcionando, fusión GPS+IMU verificada, timeline sincronizado

---

## Sprint 3: Procesamiento LiDAR Unificado ($\phi_{lidar}$ y $TRI$)

**Objetivo:** Extraer características geométricas del terreno desde LiDAR (nubes .laz y rasters .tif): inclinación transversal topográfica ($\phi_{lidar}$) e índice de rugosidad ($TRI$).

- [ ] **Tarea 3.1: Lector de nubes de puntos LAZ**
  - *Descripción:* Leer archivos .laz del CNIG PNOA con laspy. Filtrar suelo (clase 2 ASPRS). Crear índice espacial KD-tree
  - *Subtareas:*
    - [ ] Crear `scripts/lidar/laz_reader.py` clase `LAZReader(laz_path)`
    - [ ] Filtrar clasificación ASPRS (clase 2 = suelo)
    - [ ] Construir KD-tree sobre puntos del suelo
    - [ ] Método `extract_patch(x_utm, y_utm, radius_m) → np.ndarray(N,3)` — XYZ en radio
    - [ ] Método `get_bounds() → (xmin, ymin, xmax, ymax)` para selección de tiles
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint3.py::TestLAZReader::test_laz_loads` — Carga .laz sin error, ≥1000 puntos clase 2
    - [ ] `pytest scripts/tests/test_sprint3.py::TestLAZReader::test_laz_patch_nonempty` — `extract_patch` en centro del tile → ≥10 puntos
    - [ ] `pytest scripts/tests/test_sprint3.py::TestLAZReader::test_laz_crs` — CRS == EPSG:25830

- [ ] **Tarea 3.2: Lector de raster TIF (DTM)**
  - *Descripción:* Leer archivos .tif DTM con rasterio. Extraer elevaciones por UTM. Interfaz unificada con LAZ
  - *Subtareas:*
    - [ ] Crear `scripts/lidar/tif_reader.py` clase `TIFReader(tif_path)`
    - [ ] Método `extract_patch(x_utm, y_utm, radius_m) → np.ndarray(rows, cols)` — ventana de elevaciones
    - [ ] Método `get_elevation(x_utm, y_utm) → float` — elevación en punto
    - [ ] Método `get_resolution() → float` — tamaño celda en m
    - [ ] Manejar CRS y transformación afín
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTIFReader::test_tif_loads` — Carga 447-4483.tif sin error
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTIFReader::test_tif_elevation_range` — Elevaciones [400, 1200] m
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTIFReader::test_tif_patch_shape` — Patch dimensiones coherentes

- [ ] **Tarea 3.3: Interfaz unificada de terreno**
  - *Descripción:* Capa de abstracción que selecciona automáticamente LAZ o TIF según disponibilidad y coordenada
  - *Subtareas:*
    - [ ] Crear `scripts/lidar/terrain_provider.py` clase `TerrainProvider(data_dir)`
    - [ ] Auto-descubrimiento tiles .laz y .tif disponibles
    - [ ] Método `get_patch(x_utm, y_utm, radius_m, source="auto") → TerrainPatch`
    - [ ] `TerrainPatch` dataclass: elevations, resolution, crs
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainProvider::test_terrain_auto_select` — Devuelve datos TIF y LAZ
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainProvider::test_terrain_patch_valid` — Sin NaN, shape > 0

- [ ] **Tarea 3.4: Cálculo de features geométricos ($\phi_{lidar}$, $TRI$)**
  - *Descripción:* A partir de parche de terreno y Yaw ψ, calcular inclinación transversal ($\phi_{lidar}$) y rugosidad ($TRI = \sqrt{\sum(Z_{ij} - \bar{Z})^2 / N}$)
  - *Subtareas:*
    - [ ] Crear `scripts/lidar/terrain_features.py`
    - [ ] Función `compute_transverse_slope(patch, yaw_rad) → float` — Ajustar plano, proyectar gradiente perpendicular
    - [ ] Función `compute_tri(patch) → float` — RMS de elevación
    - [ ] Función `compute_all_features(patch, yaw_rad) → dict` — {phi_lidar_rad, tri, slope_along, aspect}
    - [ ] Versión vectorizada para masas de píxeles
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainFeatures::test_slope_flat` — Terreno plano → φ ≈ 0°
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainFeatures::test_slope_known` — Plano 15° → |φ − 15°| < 0.5°
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainFeatures::test_tri_flat` — Plano → TRI < 0.01
    - [ ] `pytest scripts/tests/test_sprint3.py::TestTerrainFeatures::test_tri_rough` — Rugoso → TRI > 0.1

---

## Sprint 4: Sim-to-Real (Data Augmentation con Project Chrono)

**Objetivo:** Generar datos sintéticos de vuelco seguros en simulador para balancear dataset. Validar SI simulado contra SI real en escenarios seguros.

- [ ] **Tarea 4.1: Setup del entorno Project Chrono**
  - *Descripción:* Instalar y configurar PyChrono. Wrapper de simulación programática para generar datos
  - *Subtareas:*
    - [ ] Documentar instalación PyChrono (conda-forge o compilación)
    - [ ] Crear `scripts/simulation/chrono_env.py` clase `ChronoSimEnv(vehicle_config)`
    - [ ] Modelo de vehículo rígido con parámetros Doback (M, Hg, S, Ixx)
    - [ ] Modelo de suspensión parametrizable
    - [ ] Callback para exportar: [t, x, y, v, roll, pitch, yaw, ax, ay, az, si]
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint4.py::TestChronoEnv::test_chrono_imports` — `import pychrono` sin error
    - [ ] `pytest scripts/tests/test_sprint4.py::TestChronoEnv::test_sim_flat_stable` — Simulación 10s terreno plano: roll < 2°, SI < 0.1
    - [ ] `pytest scripts/tests/test_sprint4.py::TestChronoEnv::test_sim_telemetry_columns` — 11 columnas telemetría

- [ ] **Tarea 4.2: Generador de terrenos paramétricos**
  - *Descripción:* Terrenos sintéticos con inclinación y rugosidad controladas. Importar parches reales del DTM como mallas
  - *Subtareas:*
    - [ ] Crear `scripts/simulation/terrain_gen.py`
    - [ ] `generate_tilted_plane(slope_deg, length, width, resolution)` — Superficie plana inclinada
    - [ ] `generate_rough_terrain(base_slope_deg, tri_target, seed)` — Plano + ruido gaussiano calibrado
    - [ ] `import_real_patch(terrain_patch) → chrono_terrain` — TerrainPatch → malla Chrono
    - [ ] Barrido: slopes [0°, 5°, ..., 40°] × TRI [0, 0.05, ..., 0.5] × velocidades [5, 15, 30, 50 km/h]
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint4.py::TestTerrainGen::test_tilted_plane_angle` — Generado 20° → medido 20°±0.5°
    - [ ] `pytest scripts/tests/test_sprint4.py::TestTerrainGen::test_rough_terrain_tri` — TRI ±20% del target
    - [ ] `pytest scripts/tests/test_sprint4.py::TestTerrainGen::test_real_patch_import` — Parche real importa sin error

- [ ] **Tarea 4.3: Campaña de simulación y dataset sintético**
  - *Descripción:* Ejecutar barrido masivo (~500-1000 simulaciones). Etiquetar escenarios. Combinar real+sintético con ratio vuelcos
  - *Subtareas:*
    - [ ] Crear `scripts/simulation/run_campaign.py` (CLI)
    - [ ] Ejecutar barrido completo: ~500-1000 simulaciones
    - [ ] Etiquetar: [slope_deg, tri, speed_kmh, si_max, rollover_bool]
    - [ ] Función `balance_dataset(real_df, sim_df, rollover_ratio=0.3) → combined_df`
    - [ ] Export a `datos_doback/dataset_piml.csv`
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint4.py::TestSimCampaign::test_dataset_has_rollovers` — ≥15% SI ≥ 1.0
    - [ ] `pytest scripts/tests/test_sprint4.py::TestSimCampaign::test_dataset_columns` — [speed, tri, roll_lidar, pitch_lidar, si_real, si_static, delta_si, source]
    - [ ] `pytest scripts/tests/test_sprint4.py::TestSimCampaign::test_sim_validates_real` — Escenarios seguros: |SI_sim − SI_real| < 0.15

- [ ] **Tarea 4.4: Validación Sim-to-Real**
  - *Descripción:* Validar simulaciones con parches reales reproducen SI registrados en condiciones seguras
  - *Subtareas:*
    - [ ] Seleccionar ≥10 segmentos reales con GPS+IMU+LiDAR coincidentes
    - [ ] Simular cada segmento con terreno real importado
    - [ ] RMSE y correlación SI_sim vs SI_real
    - [ ] Reporte con plots
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint4.py::TestSim2Real::test_sim2real_rmse` — RMSE(SI_sim, SI_real) < 0.2
    - [ ] `pytest scripts/tests/test_sprint4.py::TestSim2Real::test_sim2real_correlation` — Pearson r > 0.7

---

## Sprint 5: Módulo ML Predictivo ($\Delta SI$)

**Objetivo:** Entrenar modelo ML (XGBoost/RF) para predecir la perturbación dinámica ΔSI causada por velocidad y rugosidad del terreno.

- [ ] **Tarea 5.1: Feature engineering pipeline**
  - *Descripción:* Construir pipeline: datos crudos → features del modelo. Inputs: [v, TRI, φ_{lidar}, θ_{lidar}, M, suspension_type]. Target: ΔSI
  - *Subtareas:*
    - [ ] Crear `scripts/ml/feature_engineering.py`
    - [ ] `build_feature_matrix(dataset_csv) → (X, y, feature_names)`
    - [ ] Normalización StandardScaler, one-hot encoding para suspension_type
    - [ ] Split 70/15/15 (train/val/test) estratificado por source
    - [ ] EDA: distribución features, correlaciones, matriz dispersión
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint5.py::TestFeatureEng::test_feature_matrix_shape` — X: 6+ cols, y: filasX
    - [ ] `pytest scripts/tests/test_sprint5.py::TestFeatureEng::test_no_data_leakage` — Test set disjunto de train
    - [ ] `pytest scripts/tests/test_sprint5.py::TestFeatureEng::test_target_is_residual` — y ≈ si_real − si_static

- [ ] **Tarea 5.2: Entrenamiento y selección de modelo**
  - *Descripción:* Entrenar XGBoost y RF para ΔSI. Hyperparameter tuning (CV). Seleccionar mejor. Feature importance
  - *Subtareas:*
    - [ ] Crear `scripts/ml/train_model.py` (CLI)
    - [ ] XGBRegressor: n_est ∈ [100, 500], max_depth ∈ [3, 5, 8], lr ∈ [0.01, 0.05, 0.1]
    - [ ] RandomForestRegressor baseline
    - [ ] 5-fold CV, métrica RMSE+MAE
    - [ ] Save best con joblib en `scripts/ml/models/best_model.joblib`
    - [ ] Feature importance plot (SHAP o built-in)
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint5.py::TestMLTrain::test_model_rmse` — RMSE test < 0.15
    - [ ] `pytest scripts/tests/test_sprint5.py::TestMLTrain::test_model_loads` — joblib.load sin error
    - [ ] `pytest scripts/tests/test_sprint5.py::TestMLTrain::test_prediction_range` — Pred ΔSI ∈ [−1, 1]

- [ ] **Tarea 5.3: Validación PIML end-to-end**
  - *Descripción:* SI_final = SI_static + ΔSI_pred sobre test set. Comparar con SI_real del MCU. Verificar mejora vs solo física
  - *Subtareas:*
    - [ ] Crear `scripts/ml/validate_piml.py`
    - [ ] Cada muestra test: SI_est (física), ΔSI_pred (ML), SI_final
    - [ ] Comparar SI_final vs SI_real: RMSE, MAE, R²
    - [ ] Plot: SI_final vs SI_real + línea ideal y=x
    - [ ] Verificar PIML > solo-física
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint5.py::TestPIML::test_piml_beats_physics` — RMSE(SI_final, SI_real) < RMSE(SI_staticsi_real)
    - [ ] `pytest scripts/tests/test_sprint5.py::TestPIML::test_piml_r2` — R² > 0.75
    - [ ] `pytest scripts/tests/test_sprint5.py::TestPIML::test_piml_no_false_safe` — SI_real > 0.85 → SI_final > 0.7

---

## Sprint 6: Generador Masivo de Mapas 2D de Transitabilidad

**Objetivo:** Generar mapa GeoTIFF final con SI máximo por píxel, evaluando 8 direcciones Yaw a velocidad constante utilizando sliding window vectorizado.

- [ ] **Tarea 6.1: Motor de sliding window sobre DTM**
  - *Descripción:* Bucle vectorizado por cada píxel del DTM. 8 Yaws × velocidad constante. SI_max conservador. Aplicar Física + ML
  - *Subtareas:*
    - [ ] Crear `scripts/mapping/traversability_generator.py` clase `TraversabilityMapper`
    - [ ] `compute_pixel(row, col, v_kmh) → float` — SI_max de 8 direcciones
    - [ ] Rolling window / scipy.ndimage para extracción masiva
    - [ ] Por cada parche+yaw: SI_est (física) → ΔSI (ML) → SI_final = SI_est + ΔSI
    - [ ] Tomar max sobre 8 yaws como valor conservador del píxel
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint6.py::TestMapper::test_pixel_flat` — Terreno plano → SI < 0.2
    - [ ] `pytest scripts/tests/test_sprint6.py::TestMapper::test_pixel_steep` — 30° → SI > 0.7
    - [ ] `pytest scripts/tests/test_sprint6.py::TestMapper::test_8_directions` — Evalúa exacto 8 yaws

- [ ] **Tarea 6.2: Generación del GeoTIFF de salida**
  - *Descripción:* Escribir raster GeoTIFF con georeferenciación del DTM original. Banda 1: SI_max. Colormap. Metadatos
  - *Subtareas:*
    - [ ] Escritor rasterio con CRS, transform, nodata del DTM original
    - [ ] Clasificación: Verde (SI<0.4), Amarillo (0.4≤SI<0.7), Naranja (0.7≤SI<0.85), Rojo (SI≥0.85)
    - [ ] Colormap .qml o .sld para QGIS
    - [ ] Tags metadata: velocity_kmh, model, vehicle, date
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint6.py::TestGeoTIFF::test_geotiff_valid` — Rasterio abre sin error, CRS==EPSG:25830
    - [ ] `pytest scripts/tests/test_sprint6.py::TestGeoTIFF::test_geotiff_range` — Todos valores ∈ [0, 2] ∪ {nodata}
    - [ ] `pytest scripts/tests/test_sprint6.py::TestGeoTIFF::test_geotiff_dimensions` — Shape == input DTM

- [ ] **Tarea 6.3: Optimización de rendimiento**
  - *Descripción:* Procesamiento por bloques (tiling) con rasterio. Paralelización (multiprocessing). Cache de parches superpuestos. Profiling
  - *Subtareas:*
    - [ ] Procesamiento por bloques con rasterio.windows
    - [ ] Paralelización multiprocessing.Pool o concurrent.futures
    - [ ] Cache LiDAR para píxeles adyacentes
    - [ ] Barra progreso tqdm
    - [ ] Profiling cProfile → cuellos de botella
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] `pytest scripts/tests/test_sprint6.py::TestPerf::test_small_tile_time` — Tile 100×100 px < 30 s
    - [ ] `pytest scripts/tests/test_sprint6.py::TestPerf::test_tiling_consistency` — Resultado tiling == sin tiling (diff < 1e-6)

- [ ] **Tarea 6.4: CLI final y script de ejecución**
  - *Descripción:* Script CLI que orquesta todo: carga DTM → carga modelo ML → genera mapa → exporta GeoTIFF
  - *Subtareas:*
    - [ ] Crear `scripts/mapping/generate_map.py` (argparse)
    - [ ] Args: `--dtm <path>`, `--model <path>`, `--vehicle-config <path>`, `--velocity <km/h>`, `--output <path>`, `--workers <N>`
    - [ ] Logging (INFO default, DEBUG opcional)
    - [ ] Ejemplo uso en README.md
  - 🧪 *Tests de Verificación (100% Pass):*
    - [ ] CLI completo ejecuta sin error; genera output/traversability.tif
    - [ ] `pytest scripts/tests/test_sprint6.py::TestCLI::test_generate_map_cli` — `python scripts/mapping/generate_map.py --dtm ... --model ... --velocity 30 --output output/` funciona
    - [ ] `pytest scripts/tests/test_sprint6.py::TestCLI::test_cli_help` — `--help` sin error

---

## 📊 Benchmark Esperado

| Métrica | Objetivo |
|---------|----------|
| Tests Sprint 1 | 17/17 ✅ PASSED |
| Tests Sprint 2 | 8/8 ✅ PASSED |
| Tests Sprint 3 | 13/13 (pending) |
| Tests Sprint 4 | 12/12 (pending) |
| Tests Sprint 5 | 10/10 (pending) |
| Tests Sprint 6 | 11/11 (pending) |
| **Total Tests** | **81/81** |
| Tasa validación GPS parser | 100% - 1398 registros válidos parseados |
| Tasa validación IMU parser | 100% - 22773 registros a 10 Hz |
| Ángulo crítico φc | 33.8° ± medición | 
| EKF Fuction | Convergencia ±1 m con GPS parado |
| ML RMSE (ΔSI) | < 0.15 en test set |
| PIML vs Física | Mejora R² > 10% |
| Tiempo generación mapa | < 1 hora para región Madrid |

---

## 📝 Notas de Implementación

1. **CRS:** ETRS89 UTM Zone 30N (EPSG:25830) para todas las coordenadas
2. **Frecuencias reales:** GPS 1 Hz (1 muestra/seg), IMU 10 Hz (no 50 Hz como inicialmente asumido)
3. **Ground Truth:** Columna `si_mcu` del Doback MCU usado directamente (no recalculado)
4. **Simulador:** Project Chrono (PyChrono) para dinámica vehicular off-road
5. **ML Target:** Residuo ΔSI = SI_real − SI_static (no SI directo)
6. **Colormap:** Verde < 0.4 (seguro) → Rojo ≥ 0.85 (peligro de vuelco)
7. **Precisión:** Mantener doble precisión (float64) en cálculos físicos

---

## 🔄 Verificación Global del Proyecto

Checklist final para cierre de roadmap:

- [ ] `pytest scripts/tests/ -v` — 100% de tests pasan
- [ ] Output GeoTIFF se abre correctamente en QGIS con colormap de riesgo
- [ ] Ecuación $SI_{final} = SI_{estatico} + \Delta SI_{predicho}$ aplicada en cada píxel
- [ ] README.md contiene instrucciones completas de uso
- [ ] VI `PLAN_ORIGINAL.md` preserva la versión íntegra del plan inicial
- [ ] Repositorio está clean (`.gitignore` configurable) sin datos crudos masivos

---

## 🗓️ Timeline Estimado

| Sprint | Tareas | Días | Acumulado |
|--------|--------|------|-----------|
| 1 | Setup + Physics | 3-4 días | 3-4 |
| 2 | EKF | 4-5 días | 7-9 |
| 3 | LiDAR | 5-6 días | 12-15 |
| 4 | Sim-to-Real | 7-8 días | 19-23 |
| 5 | ML | 6-7 días | 25-30 |
| 6 | Mapping | 5-6 días | 30-36 |
| **TOTAL** | **6 Sprints** | **~30-36 días** | **Entregable final** |

---

**Último actualizado:** 23 de febrero de 2026  
**Versión:** 1.0 - Implementation roadmap completo  
**Estado:** ✅ Sprint 1-2 COMPLETADOS | Sprint 3-6 PENDIENTES

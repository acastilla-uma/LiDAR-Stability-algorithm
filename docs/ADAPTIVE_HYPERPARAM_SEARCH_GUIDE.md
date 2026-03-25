# Guía Completa: Búsqueda Adaptativa de Hiperparámetros

## Resumen

El script `adaptive_hyperparam_search.py` entrena modelos de regresión de forma iterativa, modificando hiperparámetros de forma aleatoria en cada ensayo (trial), hasta alcanzar un R² objetivo especificado con restricciones de estabilidad y generalización.

---

## Argumentos por Categoría

### 1. DATOS DE ENTRADA

#### `--input-glob` (default: `Doback-Data/featured/DOBACK*.csv`)
**Descripción**: Patrones glob para seleccionar archivos CSV de entrada, relativos a la raíz del repositorio.

**Uso**:
```bash
--input-glob Doback-Data/featured/DOBACK*.csv
--input-glob "Doback-Data/featured/DOBACK024*.csv" "Doback-Data/featured/DOBACK027*.csv"
```

**Cuándo modificarlo**:
- Quieres usar **un vehículo específico**: `--input-glob "Doback-Data/featured/DOBACK024*.csv"`
- Quieres usar **una fecha específica**: `--input-glob "Doback-Data/featured/*20250929*.csv"`
- Quieres usar **todos los datos**: sin cambiar (valor por defecto)
- Quieres usar **datos procesados**: `--input-glob "Doback-Data/processed-data/*.csv"`

---

#### `--contains` (default: `None`)
**Descripción**: Filtra archivos cuyo nombre contiene TODAS las subcadenas especificadas (case-insensitive).

**Uso**:
```bash
# Solo archivos que contengan "20250929" Y "seg1"
--contains 20250929 seg1

# Solo archivos del vehículo 024
--contains 024
```

**Cuándo modificarlo**:
- Necesitas una **subselección** de datos de forma temporal
- Quieres **excluir** vehículos o fechas sin cambiar `--input-glob`
- Haces un **test rápido** con datos limitados

---

#### `--max-files` (default: `0`)
**Descripción**: Si > 0, limita el número de archivos después de aplicar glob y contains. `0` = sin límite.

**Uso**:
```bash
--max-files 0          # Sin límite (todos los archivos)
--max-files 20         # Máximo 20 archivos
--max-files 1          # Un solo archivo (debug rápido)
```

**Cuándo modificarlo**:
- Estás **debugueando** y necesitas ir rápido: `--max-files 1` o `--max-files 5`
- Tienes **demasiados archivos** y quieres una muestra: `--max-files 50`
- Entrenamiento final con todos los datos: dejar valor por defecto `0`

---

#### `--max-rows-per-source` (default: `3000`)
**Descripción**: Limita filas por archivo fuente para evitar desbalance entre fuentes. `0` = sin límite.

**Ejemplo**: Si tienes 10 archivos, cada uno aporta como máximo 3000 filas.

**Uso**:
```bash
--max-rows-per-source 3000   # Hasta 3000 filas por archivo
--max-rows-per-source 0      # Sin límite
--max-rows-per-source 500    # Máximo 500 por archivo (más balanceado)
```

**Cuándo modificarlo**:
- Algunos archivos tienen **mucho más volumen** que otros → **disminuir** a `1000` o `500`
- Quieres maximizar datos: `0` (sin límite)
- Datos muy desbalanceados: `--max-rows-per-source 1000`
- Entrenamiento rápido: `--max-rows-per-source 500`

---

#### `--max-rows` (default: `0`)
**Descripción**: Si > 0, muestrea aleatoriamente este número de filas del dataset completo. `0` = sin muestreo.

**Uso**:
```bash
--max-rows 0           # Usa todas las filas (después de otros filtros)
--max-rows 100000      # Máximo 100k filas
--max-rows 50000       # Máximo 50k filas (útil si tienes millones)
```

**Cuándo modificarlo**:
- Tienes **millones de filas** y necesitas ir rápido: `--max-rows 200000`
- Haces un **test piloto**: `--max-rows 10000`
- Entrenamiento final: dejar `0` (sin muestreo)
- Memoria insuficiente: reducir a `100000` o menos

---

#### `--target-column` (default: `None`)
**Descripción**: Especifica explícitamente el nombre de la columna objetivo. `None` = auto-detección.

**Uso**:
```bash
--target-column None   # Auto-detect (busca entre omega_rad_s, gy, gz, etc.)
--target-column gz     # Explícitamente usar la columna 'gz'
--target-column omega_rad_s  # Usar omega en radianes/segundo
```

**Cuándo modificarlo**:
- Tienes **múltiples opciones de target** y quieres probar una específica
- Quieres usar **`gz`** en lugar del auto-detectado
- Auto-detección falla → especificar manualmente

---

#### `--feature-columns` (default: `None`)
**Descripción**: Lista explícita de columnas a usar como features. `None` = valores por defecto del proyecto.

**Uso**:
```bash
# Auto-detect: roll, pitch, ax, ay, az, speed_kmh, phi_lidar, tri, ruggedness
--feature-columns 

# Seleccionar features específicas
--feature-columns roll pitch ax ay speed_kmh phi_lidar

# Incluir todas las numéricas
--feature-columns
```

**Cuándo modificarlo**:
- Sospechas que **algunos features perjudican** el modelo → excluirlos
- Quieres comparar con **distinto conjunto de features**
- Incluir solo **features cinemáticas**: `--feature-columns roll pitch ax ay speed_kmh`
- Incluir solo **features de terreno**: `--feature-columns phi_lidar tri ruggedness`
- Experimentar con **subconjuntos**: prueba incremental

---

### 2. CONFIGURACIÓN DEL MODELO

#### `--model` (default: `rf`)
**Descripción**: Tipo de modelo a optimizar. Opciones: `rf` (RandomForest), `extra_trees`, `gbr` (GradientBoosting).

**Uso**:
```bash
--model rf          # RandomForest (más rápido, interpretable)
--model extra_trees # ExtraTrees (aleatorio, rápido)
--model gbr         # GradientBoosting (secuencial, potente pero lento)
```

**Características de cada modelo**:

| Modelo | Velocidad | Estabilidad | Interpretabilidad | Cuándo usar |
|--------|-----------|-------------|-------------------|------------|
| **rf** | Rápido | Media | Excelente | Por defecto, producción |
| **extra_trees** | Muy rápido | Variable | Buena | Datos ruidosos, pruebas rápidas |
| **gbr** | Lento | Alta | Pobre | Si rf no alcanza R² objetivo |

**Cuándo modificarlo**:
- Script lento con rf → probar `--model extra_trees`
- rf no alcanza R² 0.7 → probar `--model gbr` (más potente)
- Necesitas velocidad máxima → `--model extra_trees`
- Necesitas interpretabilidad → `--model rf`

---

### 3. CRITERIOS DE PARADA Y OPTIMIZACIÓN

#### `--target-r2` (default: `0.70`)
**Descripción**: Valor objetivo de R² en el set de holdout. El script para cuando lo alcanza y cumple restricciones de generalización.

**Uso**:
```bash
--target-r2 0.70  # Objetivo ambicioso (70% de varianza explicada)
--target-r2 0.60  # Objetivo moderado
--target-r2 0.50  # Objetivo bajo (30 trials suele alcanzarse)
```

**Interpretación de valores**:
- **0.50**: Modelo básico, aceptable
- **0.60**: Modelo sólido, bueno
- **0.70**: Modelo competitivo, difícil
- **0.80**: Objetivo muy ambicioso, posiblemente imposible

**Cuándo modificarlo**:
- Primera prueba con datos nuevos → `0.60`
- Producción robusta → `0.70`
- Si 120 trials no lo alcanza → bajar a `0.65` o `0.60`
- Si alcanza fácilmente en <30 trials → subir a `0.75`

---

#### `--max-trials` (default: `80`)
**Descripción**: Número máximo de ensayos antes de parar (independiente de si se alcanza target-r2).

**Uso**:
```bash
--max-trials 30          # Test rápido
--max-trials 80          # Búsqueda estándar
--max-trials 120         # Búsqueda exhaustiva
--max-trials 200         # Búsqueda muy exhaustiva (lento)
```

**Tiempo estimado** (para rf con 448k filas):
```
max-trials=30:  ~3-5 min
max-trials=80:  ~8-12 min
max-trials=120: ~15-20 min
max-trials=200: ~30-40 min
```

**Cuándo modificarlo**:
- Test de integración → `--max-trials 10` (muy rápido)
- Exploración inicial → `--max-trials 30`
- Búsqueda de producción → `--max-trials 80` o `120`
- Datos nuevos, quieres ser exhaustivo → `--max-trials 120`
- Presupuesto computacional limitado → `--max-trials 50`

---

#### `--patience` (default: `25`)
**Descripción**: Si no hay mejora en holdout R² para N trials consecutivos, detener antes de llegar a max-trials.

**Uso**:
```bash
--patience 10   # Para si 10 trials sin mejora (detiene pronto)
--patience 25   # Para si 25 trials sin mejora (estándar, más exploración)
--patience 50   # Para si 50 trials sin mejora (muy permisivo)
```

**Cuándo modificarlo**:
- Quieres que termine rápido si no hay progreso → `--patience 15`
- Quieres máxima exploración → `--patience 50`
- Configuración estándar → dejar en `25`

---

### 4. VALIDACIÓN Y RESTRICCIONES DE GENERALIZACIÓN

#### `--holdout-frac` (default: `0.20`)
**Descripción**: Fracción de datos reservada como set final de validación (holdout). El resto es train.

**Uso**:
```bash
--holdout-frac 0.20  # 20% holdout, 80% train (estándar)
--holdout-frac 0.10  # 10% holdout, 90% train (menos validación, más entrenamiento)
--holdout-frac 0.30  # 30% holdout, 70% train (más validación, menos datos para entrenar)
```

**Recomendaciones**:
- Con **millones de filas**: `0.10` (suficiente validación)
- Datos **limitados** (< 50k): `0.30` (más holdout para validación robusta)
- Estándar: `0.20` (punto medio)

**Cuándo modificarlo**:
- Tienes muchísimos datos → `0.10`
- Datos limitados → `0.30`
- Por defecto → `0.20`

---

#### `--n-splits` (default: `5`)
**Descripción**: Número de folds para validación cruzada (K-Fold) en el set de entrenamiento.

**Uso**:
```bash
--n-splits 3   # 3-fold CV (rápido, menos validación)
--n-splits 5   # 5-fold CV (estándar, balance)
--n-splits 10  # 10-fold CV (más validación, más lento)
```

**Tiempo y validación**:
- `3`: Rápido (~30% más rápido), menos validación
- `5`: Estándar, buen balance
- `10`: Validación robusta pero +50% más lento

**Cuándo modificarlo**:
- Test rápido → `--n-splits 3`
- Datos limitados → `--n-splits 10` (más validación con menos datos)
- Producción → `--n-splits 5` (estándar)

---

#### `--max-generalization-gap` (default: `0.08`)
**Descripción**: Máximo gap permitido entre CV R² y holdout R². Control de sobreajuste.

**Gap = CV R² - Holdout R²**

Si CV dice R²=0.75 pero holdout es 0.70, gap=0.05 (aceptable si límite es 0.08).

**Uso**:
```bash
--max-generalization-gap 0.05  # Muy estricto (solo modelos excelentes)
--max-generalization-gap 0.08  # Estándar (balance)
--max-generalization-gap 0.12  # Permisivo (permite cierto sobreajuste)
```

**Interpretación**:
- Gap **pequeño** (< 0.05): Modelo generaliza excelentemente
- Gap **moderado** (0.05-0.10): Aceptable
- Gap **grande** (> 0.10): Señal de sobreajuste

**Cuándo modificarlo**:
- Datos limpios, mucha muestra → `0.05` (estricto)
- Condiciones normales → `0.08` (estándar)
- Datos ruidosos o limitados → `0.12` (permisivo)
- Si no alcanza target-r2 → aumentar a `0.10` (menos restricción)

---

#### `--max-r2-std` (default: `0.08`)
**Descripción**: Máxima desviación estándar de R² entre folds. Control de inestabilidad.

Si R² en los 5 folds es [0.70, 0.71, 0.69, 0.72, 0.68], std=0.015 (muy estable).

**Uso**:
```bash
--max-r2-std 0.05  # Muy estricto (modelos muy consistentes)
--max-r2-std 0.08  # Estándar (aceptable variabilidad)
--max-r2-std 0.15  # Permisivo (permite inestabilidad)
```

**Interpretación**:
- Std **baja** (< 0.05): Modelo muy consistente en todos los folds
- Std **moderada** (0.05-0.10): Aceptable
- Std **alta** (> 0.10): Señal de inestabilidad

**Cuándo modificarlo**:
- Datos muy estables → `0.05` (estricto)
- Condiciones normales → `0.08` (estándar)
- Datos ruidosos → `0.12` (permisivo)
- Si no alcanza target-r2 → aumentar a `0.12`

---

### 5. REPRODUCIBILIDAD Y SALIDA

#### `--random-state` (default: `42`)
**Descripción**: Semilla para generador de números aleatorios. Asegura reproducibilidad.

**Uso**:
```bash
--random-state 42   # Reproducible (mismo resultado siempre)
--random-state 123  # Reproducible con otra semilla
--random-state 99   # Reproducible
```

**Cuándo modificarlo**:
- Necesitas **reproducibilidad exacta**: mantén el mismo valor
- Quieres probar **múltiples búsquedas independientes**: usa `42`, `123`, `456` en tres ejecuciones
- Entorno de producción: mantén el mismo valor

---

#### `--output-dir` (default: `output/models`)
**Descripción**: Directorio donde guardar modelos, historial y leaderboard (relativo a raíz del repo).

**Uso**:
```bash
--output-dir output/models                    # Ubicación por defecto
--output-dir output/experiments/exp_001       # Experimento específico
--output-dir output/final_runs                # Runs definitivos
```

**Cuándo modificarlo**:
- Quieres organizar por **experimentos**: `output/experiments/exp_lidar_only`
- Quieres separar **runs de producción**: `output/production`
- Por defecto → `output/models`

---

#### `--prefix` (default: `adaptive_w_model`)
**Descripción**: Prefijo de archivos de salida. Los archivos serán: `{prefix}_{model}_best.joblib`, etc.

**Uso**:
```bash
--prefix adaptive_w_model        # Por defecto
--prefix rf_lidar_analysis       # Descriptivo
--prefix final_production_r2_70  # Indicar versión/target
--prefix exp001_rf               # Experimento numbered
```

**Salida generada**:
```
output/models/
├── {prefix}_rf_best.joblib           # Modelo final
├── {prefix}_rf_history.json          # Historial de todos los trials
└── {prefix}_rf_leaderboard.csv       # Top trials ordenado por R²
```

**Cuándo modificarlo**:
- Hacer múltiples **experimentos en paralelo**: usa prefijos diferentes
- Documentar el **objetivo**: `rf_target_r2_70`
- Versionar ejecuciones: `exp_20250325_001`

---

## Ejemplos de Uso Práctico

### Ejemplo 1: Test Rápido (2 minutos)
```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --max-trials 10 \
  --max-files 5 \
  --max-rows 50000 \
  --target-r2 0.60 \
  --prefix quick_test
```

### Ejemplo 2: Búsqueda Estándar (15 minutos)
```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --max-trials 80 \
  --patience 25 \
  --target-r2 0.70 \
  --max-generalization-gap 0.08 \
  --prefix standard_search
```

### Ejemplo 3: Vehículo Específico (reducido)
```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --input-glob "Doback-Data/featured/DOBACK024*.csv" \
  --max-rows-per-source 1000 \
  --max-trials 60 \
  --target-r2 0.65 \
  --prefix vehicle_024_analysis
```

### Ejemplo 4: Exhaustivo (muy lento)
```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model gbr \
  --max-trials 150 \
  --patience 50 \
  --n-splits 10 \
  --holdout-frac 0.15 \
  --max-generalization-gap 0.05 \
  --max-r2-std 0.05 \
  --prefix exhaustive_gbr_search
```

### Ejemplo 5: Solo Features Cinemáticas
```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --feature-columns roll pitch ax ay az speed_kmh \
  --max-trials 60 \
  --target-r2 0.60 \
  --prefix kinematic_only
```

---

## Matriz de Decisión: Qué Modificar Según Caso

| Situación | Cambios Recomendados |
|-----------|----------------------|
| **Primera prueba con datos nuevos** | `--max-trials 30 --target-r2 0.60 --max-files 10` |
| **Modelo tarda demasiado** | `--max-trials 50 --max-rows 100000 --model extra_trees` |
| **No alcanza R² 0.70** | `--target-r2 0.65` o `--model gbr` o `--max-generalization-gap 0.10` |
| **Sospechas sobreajuste** | Disminuir `--max-generalization-gap 0.05` y `--max-r2-std 0.05` |
| **Datos limitados (< 50k)** | `--holdout-frac 0.30 --n-splits 10 --max-rows-per-source 0` |
| **Datos masivos (> 1M)** | `--max-rows 300000 --max-rows-per-source 1000 --max-files 100` |
| **Quieres reproducibilidad exacta** | Mantén `--random-state 42` |
| **Múltiples ejecuciones independientes** | Usa `--prefix exp_01`, `--prefix exp_02`, etc. |
| **Debug rápido** | `--max-trials 5 --max-files 1 --max-rows 10000` |
| **Producción robusta** | `--max-trials 120 --patience 30 --max-generalization-gap 0.08` |

---

## Interpretación de Salidas

### Archivo: `{prefix}_rf_history.json`
Contiene todos los trials. Ejemplo de entrada:
```json
{
  "trial": 1,
  "cv_r2_mean": 0.642,
  "cv_r2_std": 0.028,
  "holdout_r2": 0.635,
  "generalization_gap": 0.007,
  "params": { "rf_n_estimators": 450, "rf_min_samples_leaf": 3, ... }
}
```

**Interpreta**:
- Si `generalization_gap` > `max_generalization_gap`: modelo rechazado
- Si `cv_r2_std` > `max_r2_std`: modelo inestable, rechazado
- Si `holdout_r2` alcanza target-r2: **ganador** (si cumple restricciones)

---

### Archivo: `{prefix}_rf_leaderboard.csv`
Top trials ordenados por `holdout_r2`. Úsalo para:
- Ver los 3-5 mejores modelos
- Identificar hiperparámetros que funcionan
- Detectar patrones (p. ej., n_estimators siempre alto)

---

## Checklist Antes de Ejecutar

- [ ] Tienes datos en `Doback-Data/featured` o especificas `--input-glob` correcto
- [ ] Cantidad de memoria razonable (si usas `--max-rows` muy alto, puede fallar)
- [ ] Tiempo disponible (estima con la tabla de tiempo de arriba)
- [ ] Objetivo R² realista (0.60-0.70 es estándar con estos datos)
- [ ] Output dir es escribible y no conflictúa con otros experimentos

---

## Notas Finales

- **Reproducibilidad**: Mantén `--random-state` constante si necesitas repetir
- **Iterativo**: Comienza con test rápido, luego aumenta max-trials
- **Monitoreo**: Observa los primeros trials; si cv_r2_mean < 0.50 desde el inicio, los datos tienen problemas
- **Interpretación**: Un gap grande NO significa que el modelo sea malo; significa que se ajusta bien al train y generaliza menos

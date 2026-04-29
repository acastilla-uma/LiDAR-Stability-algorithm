# Guía de `adaptive_hyperparam_search.py`

Esta guía explica cómo usar el script, cómo funciona internamente y qué significa cada comando disponible hoy.

El archivo de referencia es [src/lidar_stability/ml/adaptive_hyperparam_search.py](../../src/lidar_stability/ml/adaptive_hyperparam_search.py).

## Qué hace el script

`adaptive_hyperparam_search.py` entrena modelos de regresión con búsqueda adaptativa de hiperparámetros usando Optuna.

El flujo general es:

1. Busca los CSV de entrada.
2. Carga y limpia los datos.
3. Divide el dataset en entrenamiento y holdout.
4. Ejecuta validación cruzada en cada trial.
5. Prueba hiperparámetros con Optuna.
6. Guarda el mejor modelo.
7. Exporta historial, leaderboard y un reporte Markdown con explicabilidad.

## Cómo ejecutarlo

Ejemplo básico:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py
```

Ejemplo con parámetros más controlados:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --input-glob "Doback-Data/featured/DOBACK024*.csv" \
  --model rf \
  --target-r2 0.70 \
  --max-trials 80 \
  --n-splits 5 \
  --holdout-frac 0.20 \
  --output-dir output/models \
  --prefix adaptive_w_model
```

## Qué hace internamente

### 1. Selección de archivos

El script toma `--input-glob`, busca los archivos desde la raíz del repositorio y filtra con `--contains` si lo indicas.

### 2. Carga y preparación

- Lee todos los CSV seleccionados.
- Agrega una columna interna `__source_file` para conservar el origen de cada fila.
- Construye el dataset de entrenamiento con `build_w_training_dataset`.
- Usa `--feature-columns` si se indican.
- Si no se indican features, usa `DEFAULT_FEATURE_COLUMNS`.

### 3. División train/holdout

- Divide el dataset en entrenamiento y holdout con `--holdout-frac`.
- La separación se hace por grupos de archivo fuente cuando esa columna está disponible.
- Esto reduce fuga de información entre train y holdout.

### 4. Búsqueda Optuna

Para cada trial:

- Se proponen hiperparámetros según `--model`.
- Se entrena un modelo en cada fold de CV.
- Se calcula R2, RMSE y MAE por fold.
- Si el trial es flojo, el pruner puede detenerlo antes.
- Al final del trial se entrena un modelo final sobre todo el entrenamiento y se evalúa en holdout.

### 5. Selección del mejor trial

El mejor trial se elige con estas prioridades:

- Primero, trials que cumplan las restricciones.
- Luego, mayor `holdout_r2`.
- Si hay empate, mayor `cv_r2_mean`.
- Si todavía hay empate, el trial más temprano gana.

### 6. Artefactos generados

El script guarda en `--output-dir`:

- `*_best.joblib`: modelo entrenado final.
- `*_history.json`: historial completo del estudio.
- `*_leaderboard.csv`: ranking de trials.
- `*_leaderboard.json`: leaderboard con más contexto.
- `*_report.md`: guía de explicabilidad y diagnóstico.
- `*_study.sqlite3`: estudio Optuna persistido.

## Qué incluye el reporte Markdown

El archivo `*_report.md` se guarda en la misma carpeta del modelo y resume:

- evolución de los trials,
- métricas por fold,
- generalization gap,
- residuales del holdout,
- importancia de variables del modelo,
- permutation importance,
- notas de posible fallo o sobreajuste.

## Referencia de comandos

### `--input-glob`

**Qué hace:** patrones glob para encontrar archivos CSV de entrada.

**Por defecto:** `Doback-Data/featured/DOBACK*.csv`

**Ejemplos:**

```bash
--input-glob Doback-Data/featured/DOBACK*.csv
--input-glob "Doback-Data/featured/DOBACK024*.csv"
--input-glob "Doback-Data/featured/DOBACK024*.csv" "Doback-Data/featured/DOBACK027*.csv"
```

**Cuándo usarlo:**

- Si quieres trabajar con un vehículo concreto.
- Si quieres comparar fechas o segmentos concretos.
- Si quieres limitar el conjunto de datos sin tocar el código.

---

### `--contains`

**Qué hace:** filtra por subcadenas del nombre del archivo. Todas las subcadenas deben aparecer.

**Por defecto:** `None`

**Ejemplos:**

```bash
--contains 024
--contains 20251007 seg24
```

**Cuándo usarlo:**

- Para afinar aún más `--input-glob`.
- Para pruebas rápidas con una parte concreta del dataset.

---

### `--model`

**Qué hace:** elige la familia de modelo a optimizar.

**Opciones:**

- `rf`: `RandomForestRegressor`
- `extra_trees`: `ExtraTreesRegressor`
- `gbr`: `GradientBoostingRegressor`

**Por defecto:** `rf`

**Ejemplos:**

```bash
--model rf
--model extra_trees
--model gbr
```

**Cómo pensar cada opción:**

- `rf`: balance entre calidad, estabilidad e interpretabilidad.
- `extra_trees`: suele explorar rápido y ser robusto al ruido.
- `gbr`: puede capturar patrones finos, pero suele ser más sensible y lento.

---

### `--target-r2`

**Qué hace:** define el objetivo mínimo de R2 en holdout para parar antes de agotar los trials.

**Por defecto:** `0.70`

**Ejemplos:**

```bash
--target-r2 0.60
--target-r2 0.70
--target-r2 0.80
```

**Qué significa:**

- Si el mejor trial alcanza ese valor en holdout y además cumple las restricciones, el script puede detenerse.
- Es una forma de fijar un criterio de éxito práctico.

---

### `--max-trials`

**Qué hace:** máximo de trials que Optuna intentará.

**Por defecto:** `80`

**Ejemplos:**

```bash
--max-trials 10
--max-trials 30
--max-trials 80
--max-trials 120
```

**Cuándo tocarlo:**

- `10` o `30` para pruebas rápidas.
- `80` como búsqueda estándar.
- `120` si quieres explorar más y tienes tiempo de cómputo.

---

### `--patience`

**Qué hace:** detiene la búsqueda si no hay mejora en `holdout_r2` durante N trials completos consecutivos.

**Por defecto:** `25`

**Ejemplos:**

```bash
--patience 10
--patience 25
--patience 50
```

**Interpretación:**

- Valor bajo: para antes.
- Valor alto: explora más.

---

### `--n-splits`

**Qué hace:** número de folds en la validación cruzada.

**Por defecto:** `5`

**Ejemplos:**

```bash
--n-splits 3
--n-splits 5
--n-splits 10
```

**Efecto práctico:**

- Más folds = evaluación más estable, pero más lenta.
- Menos folds = más rápida, pero menos robusta.

---

### `--n-jobs`

**Qué hace:** número de procesos o hilos para modelos que lo soportan.

**Por defecto:** `-1`

**Ejemplos:**

```bash
--n-jobs -1
--n-jobs 4
--n-jobs 1
```

**Cuándo usarlo:**

- `-1` para usar todos los cores disponibles.
- `1` si quieres evitar saturar la máquina o reproducir pruebas más controladas.

---

### `--holdout-frac`

**Qué hace:** porcentaje de datos reservado para holdout final.

**Por defecto:** `0.20`

**Ejemplos:**

```bash
--holdout-frac 0.10
--holdout-frac 0.20
--holdout-frac 0.30
```

**Cuándo cambiarlo:**

- `0.10` si tienes muchos datos.
- `0.30` si necesitas una validación final más fuerte.

---

### `--max-generalization-gap`

**Qué hace:** limita la diferencia entre `cv_r2_mean` y `holdout_r2`.

**Por defecto:** `0.08`

**Ejemplos:**

```bash
--max-generalization-gap 0.05
--max-generalization-gap 0.08
--max-generalization-gap 0.12
```

**Qué significa:**

- Gap pequeño: el modelo generaliza mejor.
- Gap grande: posible sobreajuste.

---

### `--max-r2-std`

**Qué hace:** limita la dispersión del R2 entre folds.

**Por defecto:** `0.08`

**Ejemplos:**

```bash
--max-r2-std 0.05
--max-r2-std 0.08
--max-r2-std 0.15
```

**Qué significa:**

- Valor bajo: el modelo es más consistente.
- Valor alto: el comportamiento cambia mucho entre folds.

---

### `--target-column`

**Qué hace:** nombre de la columna objetivo.

**Por defecto:** `gy`

**Ejemplo:**

```bash
--target-column gy
```

**Nota:** en este flujo el target esperado es `gy`.

---

### `--feature-columns`

**Qué hace:** define explícitamente las columnas de entrada que el modelo usará.

**Por defecto:** `None`, lo que significa usar las features por defecto del proyecto.

**Ejemplos:**

```bash
--feature-columns roll pitch ax ay speed_kmh
--feature-columns phi_lidar tri ruggedness
```

**Cuándo usarlo:**

- Si quieres probar un subconjunto concreto.
- Si sospechas que algunas variables meten ruido.
- Si quieres comparar familias de features.

---

### `--random-state`

**Qué hace:** semilla de reproducibilidad.

**Por defecto:** `42`

**Ejemplos:**

```bash
--random-state 42
--random-state 123
```

**Cuándo tocarlo:**

- Si quieres reproducir exactamente un experimento.
- Si quieres varias corridas independientes, cambia la semilla.

---

### `--output-dir`

**Qué hace:** carpeta donde se guardan modelo, historial, leaderboard y reporte.

**Por defecto:** `output/models`

**Ejemplos:**

```bash
--output-dir output/models
--output-dir output/experiments/run_01
```

---

### `--prefix`

**Qué hace:** prefijo usado para los archivos generados.

**Por defecto:** `adaptive_w_model`

**Ejemplos:**

```bash
--prefix adaptive_w_model
--prefix exp_a
```

**Resultado:**

Si usas `--prefix exp_a` y `--model rf`, obtendrás archivos como:

- `exp_a_rf_best.joblib`
- `exp_a_rf_history.json`
- `exp_a_rf_leaderboard.csv`
- `exp_a_rf_report.md`

---

### `--sampler`

**Qué hace:** elige el generador de sugerencias de Optuna.

**Opciones:**

- `tpe`: `TPESampler`
- `random`: muestreo aleatorio puro

**Por defecto:** `tpe`

**Ejemplos:**

```bash
--sampler tpe
--sampler random
```

**Cuándo usarlo:**

- `tpe` para una búsqueda más inteligente.
- `random` si quieres una línea base o depurar.

---

### `--pruner`

**Qué hace:** decide si Optuna corta trials malos antes de terminar.

**Opciones:**

- `median`: usa `MedianPruner`
- `none`: desactiva pruning

**Por defecto:** `median`

**Ejemplos:**

```bash
--pruner median
--pruner none
```

**Cuándo usarlo:**

- `median` si quieres ahorrar tiempo.
- `none` si prefieres ejecutar todos los folds de todos los trials.

---

### `--startup-trials`

**Qué hace:** número de trials iniciales antes de que el sampler TPE y el pruner se apoyen en historial suficiente.

**Por defecto:** `10`

**Ejemplos:**

```bash
--startup-trials 5
--startup-trials 10
--startup-trials 20
```

**Cuándo tocarlo:**

- Más bajo para empezar a optimizar antes.
- Más alto si quieres un arranque más conservador.

---

### `--study-name`

**Qué hace:** nombre lógico del estudio Optuna.

**Por defecto:** `None`

**Ejemplos:**

```bash
--study-name exp_001_rf
--study-name vehicle024_search
```

**Cuándo usarlo:**

- Cuando quieras rastrear varias búsquedas.
- Cuando quieras reanudar una ejecución concreta.

---

### `--study-storage`

**Qué hace:** ruta del archivo SQLite que persiste el estudio Optuna.

**Por defecto:** `None`, lo que hace que se cree algo como:

`<output-dir>/<prefix>_<model>_study.sqlite3`

**Ejemplos:**

```bash
--study-storage output/models/adaptive_w_model_rf_study.sqlite3
--study-storage output/experiments/run_01/study.sqlite3
```

**Cuándo usarlo:**

- Si quieres guardar el estudio en una ruta concreta.
- Si quieres compartir o reanudar el estudio después.

---

### `--resume / --no-resume`

**Qué hace:** controla si el script reanuda un estudio existente o crea uno nuevo.

**Por defecto:** `--resume`

**Ejemplos:**

```bash
--resume
--no-resume
```

**Cuándo usarlo:**

- `--resume` para continuar una búsqueda interrumpida.
- `--no-resume` para forzar una búsqueda nueva.

## Cómo interpretar los resultados

### `history.json`

Guarda:

- configuración usada,
- número de trials,
- mejor trial,
- trials completados,
- trials pruned,
- ruta del modelo,
- ruta del reporte.

### `leaderboard.csv`

Es una tabla ordenada por rendimiento. Sirve para comparar los trials de forma rápida.

### `leaderboard.json`

Contiene lo mismo que el leaderboard, pero con más contexto por trial.

### `report.md`

Es el resumen humano. Úsalo para responder:

- qué aprendió el modelo,
- qué variables pesan más,
- si el modelo generaliza bien,
- en qué falló,
- si hay señales de sobreajuste.

## Qué mirar si el modelo falla

1. `holdout_r2` muy bajo.
2. `generalization_gap` grande.
3. `cv_r2_std` alta.
4. Residuales del holdout muy dispersos.
5. Variables importantes poco estables entre trials.

## Recomendación de uso rápido

Para una primera prueba:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --max-trials 30 \
  --patience 10 \
  --n-splits 5 \
  --target-r2 0.60
```

Para una corrida más seria:

```bash
python src/lidar_stability/ml/adaptive_hyperparam_search.py \
  --model rf \
  --max-trials 80 \
  --patience 25 \
  --n-splits 5 \
  --target-r2 0.70 \
  --sampler tpe \
  --pruner median
```

## Resumen corto

Si solo quieres recordar lo importante:

- `--input-glob`: qué archivos leer.
- `--model`: qué familia de modelo optimizar.
- `--target-r2`: cuándo considerar que el objetivo se cumplió.
- `--max-trials`: cuánto explorar.
- `--patience`: cuándo parar por falta de mejora.
- `--n-splits`: cuánta validación cruzada usar.
- `--holdout-frac`: cuánto reservar para evaluación final.
- `--sampler` y `--pruner`: cómo explora Optuna.
- `--feature-columns`: qué variables usar.
- `--output-dir` y `--prefix`: dónde y con qué nombre guardar resultados.
- `report.md`: el resumen explicable del aprendizaje y de los fallos.

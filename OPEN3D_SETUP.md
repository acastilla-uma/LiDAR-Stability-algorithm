# Visualización 3D con Open3D - Guía de Instalación Local

Este script permite visualizar segmentos LiDAR con máxima densidad de puntos (hasta 1M de puntos) usando Open3D con aceleración GPU.

## Requisitos Previos

✅ **Sistema Operativo**: Windows, macOS o Linux
✅ **GPU**: Recomendado (NVIDIA/AMD/Intel Arc)
✅ **RAM**: Mínimo 8GB
✅ **Python**: 3.10 o superior

## Instalación

### 1. Instalar Open3D

#### Opción A: pip (recomendado)
```bash
pip install open3d
```

#### Opción B: conda (si tienes problemas con pip)
```bash
conda install -c open3d-admin open3d
```

#### Opción C: Desde fuente (para máxima optimización GPU)
```bash
git clone https://github.com/isl-org/Open3D.git
cd Open3D
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 2. Verificar instalación

```bash
python -c "import open3d as o3d; print(f'Open3D {o3d.__version__} ✅')"
```

Deberías ver: `Open3D 0.X.X ✅`

## Uso

### Ejecución Básica (500k puntos por defecto)

```bash
python src/lidar_stability/visualization/visualize_3d_open3d.py --base "DOBACK024_20250929"
```

### Con Máxima Densidad (1M de puntos)

```bash
python src/lidar_stability/visualization/visualize_3d_open3d.py \
  --base "DOBACK024_20250929" \
  --points-sample 1000000
```

### Con Todas las Opciones

```bash
python src/lidar_stability/visualization/visualize_3d_open3d.py \
  --base "DOBACK024_20250929" \
  --points-sample 1000000 \
  --no-ground-filter \
  --padding 200
```

## Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--base` | str | ✓ requerido | Nombre base del segmento (ej: DOBACK024_20250929) |
| `--points-sample` | int | 1,000,000 | Máximo número de puntos a visualizar |
| `--stability-col` | str | "si" | Nombre de la columna de estabilidad |
| `--no-ground-filter` | flag | False | Incluir todos los puntos (no filtrar suelo) |
| `--padding` | float | 100 | Margen alrededor de la ruta en metros |
| `--mapmatch-dir` | str | auto | Directorio con archivos map-matched |
| `--laz-dir` | str | auto | Directorio con archivos LAZ |

## Controles en la Ventana 3D

### Navegación
- **Botón Izquierdo + Mover**: Rotar vista
- **Botón Derecho + Mover**: Trasladar (panorama)
- **Rueda Ratón**: Zoom in/out
- **Scroll**: Acercar/alejar

### Teclas
- `Z`: Resetear vista a posición inicial
- `H`: Mostrar ayuda
- `Esc` / `Q`: Cerrar visualizador
- `S`: Captura de pantalla (guarda en disco)

## Rendimiento

### Recomendaciones por Cantidad de Puntos

| Puntos | RAM | GPU | Ventana | Interacción |
|--------|-----|-----|---------|-------------|
| 100k | 2GB | No | Fluida | Excelente |
| 500k | 4GB | Recomendado | Fluida | Buena |
| 1M | 8GB | **Requerida** | Normal | Aceptable |
| 2M+ | 16GB+ | **Esencial** | Lenta | Entrecortada |

### Optimización GPU

Si tu GPU no se usa automáticamente, intenta:

**NVIDIA (CUDA):**
```bash
pip install open3d-ml
```

**AMD (HIP):**
```bash
pip install open3d[amd]
```

**Intel Arc:**
```bash
pip install open3d[oneapi]
```

## Solución de Problemas

### Error: "No module named 'open3d'"
```bash
pip install --upgrade open3d
```

### Error: "CUDA/GPU not available"
Open3D puede funcionar sin GPU (más lentamente). Para forzar CPU:
```python
import open3d as o3d
o3d.core.cuda.is_available = lambda: False
```

### Visualización muy lenta
- Reduce `--points-sample` a 500k o 100k
- Cierra otras aplicaciones pesadas
- Verifica que GPU esté disponible

### Puntos no visibles
- Asegúrate que existen archivos LAZ en: `LiDAR-Maps/cnig/`
- Verifica que CSV de segmentos exista en: `Doback-Data/featured/`
- Intenta sin filtrado: `--no-ground-filter`

### Bloqueo de la ventana
- Esto es normal durante la carga de 1M puntos (~30-60 segundos)
- No cierres la ventana, espera a que termine

## Exportar Captura

Durante la visualización, presiona `S` para guardar una captura de pantalla PNG en el directorio actual.

## Información Adicional

- Estado de GPU: Se mostrará en logs al inicio
- Estadísticas de carga: Se imprimirán en terminal
- Tiempo de renderizado: Indicado al final de la carga

## Ejemplos de Uso Completo

```bash
# Visualizar con máxima calidad
cd /ruta/a/LiDAR-Stability-algorithm
python src/lidar_stability/visualization/visualize_3d_open3d.py \
  --base "DOBACK024_20250929" \
  --points-sample 1000000 \
  --padding 150

# Ver todos los puntos sin filtrado
python src/lidar_stability/visualization/visualize_3d_open3d.py \
  --base "DOBACK024_20250929" \
  --points-sample 1000000 \
  --no-ground-filter
```

---

## Soporte

Si experimentas problemas:
1. Verifica Python ≥ 3.10: `python --version`
2. Actualiza dependencias: `pip install --upgrade open3d`
3. Comprueba ruta de archivos LAZ: `ls LiDAR-Maps/cnig/ | head -5`
4. Revisa logs en terminal para mensajes de error específicos

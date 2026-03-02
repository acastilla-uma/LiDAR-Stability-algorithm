# Sprint 1: Batch Processing & Route Visualization ✅

**Status:** COMPLETED  
**Date Completed:** February 26, 2026  
**Purpose:** Production-grade batch processing of DOBACK GPS + stability sensor data

---

## 🎯 Objectives

Sprint 1 includes production-ready batch processing and visualization:
1. Match GPS and stability measurements by timestamp
2. Filter outliers and anomalies
3. Segment routes based on spatial gaps
4. Generate clean, analysis-ready CSV datasets
5. Interactive visualization of processed routes

This completes the data acquisition and preparation layer: **raw sensor logs** → **clean segmented datasets** → **ready for Sprint 2 (EKF fusion)**.

---

## 📦 Deliverables

### 1. Route Processing Pipeline
**File:** `Scripts/parsers/batch_processor.py`  
**Purpose:** Batch process all GPS + stability pairs, output segmented CSVs

**Features:**
- ✅ Temporal matching: GPS 1 Hz + Stability 10 Hz via `pd.merge_asof`
- ✅ Coordinate validation: Spanish Peninsula bounds (35-45°N, -10 to 5°E)
- ✅ UTM conversion: EPSG:25830 (UTM Zone 30N) for distance calculations
- ✅ Anomaly filtering:
  - GPS jumps >100m between consecutive points
  - Isolated points (<1 neighbor within 50m in ±2 point window)
- ✅ Route segmentation: Splits on gaps >1000m (configurable)
- ✅ Size filtering: Discards segments with <10 points

**Input:**
```
Doback-Data/
├── GPS/
│   ├── GPS_DOBACK023_20251012.txt
│   ├── GPS_DOBACK024_20251005.txt
│   └── ...
└── Stability/
    ├── ESTABILIDAD_DOBACK023_20251012.txt
    ├── ESTABILIDAD_DOBACK024_20251005.txt
    └── ...
```

**Output:**
```
Doback-Data/processed data/
├── DOBACK023_20251012.csv           # Single continuous route
├── DOBACK024_20251005_seg1.csv      # Multi-segment route
├── DOBACK024_20251005_seg2.csv
├── ...
└── processing_report.txt            # Processing statistics
```

**Usage:**
```bash
python Scripts/parsers/batch_processor.py \
  --tolerance-seconds 1.0 \
  --max-gap-meters 1000
```

---

### 2. Interactive Route Visualization
**File:** `Scripts/parsers/route_visualizer.py`  
**Purpose:** Generate interactive HTML maps with SI-based color coding

**Features:**
- 🗺️ Folium/OpenStreetMap interactive maps
- 🎨 Gradient color scale: Red (SI=0) → Yellow (SI=0.5) → Green (SI=1)
- 📊 Dynamic legend with SI range, point count, segment info
- 🔍 Multi-segment support: overlay multiple routes on one map
- 🌐 Auto-open in browser
- 🔎 Pattern-based file discovery: `DOBACK024_20251005` → finds all `_segN.csv` automatically

**Usage:**
```bash
# Visualize all segments of a route
python Scripts/parsers/route_visualizer.py \
  "Doback-Data/processed data/DOBACK024_20251005"

# Visualize specific segments
python Scripts/parsers/route_visualizer.py \
  seg1.csv seg2.csv seg3.csv \
  --output output/custom_map.html
```

**Output:** `output/mapa_ruta_si.html` (interactive HTML map)

---

## 📊 Processing Statistics

**Data Processed:**
- **Devices:** DOBACK023, DOBACK024, DOBACK027, DOBACK028
- **Date Range:** September 2025 - February 2026
- **Total Routes:** ~150+ GPS/Stability pairs
- **Valid Routes:** ~60% (remaining have no matched rows due to timestamp misalignment)
- **Total Segments:** ~800+ (after gap-based splitting)
- **Total Points:** ~2.5M+ matched GPS+stability records

**Example Session:** DOBACK024_20251005
```
15 segments total, 12 saved, 3 discarded (<10 pts)
seg1: 1,501 rows (SI range: 0.860 - 0.910)
seg2: 509 rows (SI range: 0.870 - 0.940)
seg3: 1,417 rows (SI range: 0.580 - 0.930)
...
```

---

## 🧪 Validation

### Data Quality Checks:
```python
import pandas as pd
import glob

# Load all processed CSVs
files = glob.glob("Doback-Data/processed data/DOBACK*.csv")
print(f"Total files: {len(files)}")

for f in files[:5]:  # Sample check
    df = pd.read_csv(f)
    assert all(col in df.columns for col in ['timestamp', 'lat', 'lon', 'si'])
    assert df['si'].between(0, 2).all(), f"Invalid SI in {f}"
    assert df['lat'].between(35, 45).all()
    print(f"✓ {f.split('/')[-1]}: {len(df)} points, SI [{df['si'].min():.2f}, {df['si'].max():.2f}]")
```

### Visual Inspection:
- Maps generated with color-coded routes reveal spatial patterns in stability
- Red segments indicate problematic terrain (SI < 0.7)
- Green segments show stable terrain (SI ≥ 1.0)

---

## 🔗 Integration with Project Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Sprint 1: Data Processing & Physics                        │
│  ────────────────────────────                               │
│  Input:  GPS_*.txt + ESTABILIDAD_*.txt (raw sensor logs)   │
│  Output: DOBACK*_seg*.csv (clean, matched, segmented)      │
│  batch_processor.py  → Matching + segmentación             │
│  route_visualizer.py → Mapas interactivos SI               │
│  stability_engine.py → Calculate SI_static                 │
│  ground_truth.py     → Generate ΔSI targets                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Sprint 2: Sensor Fusion (EKF)                              │
│  ───────────────────────────                                │
│  ekf/ekf_fusion.py → Fuse GPS 1Hz + IMU 10Hz               │
│  ekf/time_sync.py  → Synchronize timestamps                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Sprint 3: LiDAR Integration                                │
│  ────────────────────────                                   │
│  lidar/laz_reader.py → Point cloud processing              │
│  lidar/terrain_features.py → φ_lidar, TRI extraction       │
└─────────────────────────────────────────────────────────────┘
```

**Note:** Sprint 1 batch processing produces analysis-ready CSVs directly from raw logs. These outputs feed the physics and ground-truth steps without extra preprocessing steps.

---

## 🛠️ Technical Details

### Timestamp Matching Strategy:
- GPS: `datetime.strptime(fecha + hora_gps)` → UTC timestamp
- Stability: `base_datetime + timedelta(microseconds=timeantwifi)`
- Merging: `pd.merge_asof(tolerance=1.0s)` → nearest neighbor within 1 second

### Spatial Filtering:
- **Coordinate validation:** 35° ≤ lat ≤ 45°, -10° ≤ lon ≤ 5° (Spain)
- **Altitude range:** 0 < alt < 3000 m
- **Speed limit:** 0 ≤ v < 200 km/h
- **Jump detection:** Distance >100m between consecutive points (using UTM)

### Segmentation Logic:
```python
def split_into_segments(df, max_gap_meters=1000):
    distances = sqrt(diff(x_utm)^2 + diff(y_utm)^2)
    gap_indices = where(distances > max_gap_meters)
    # Split dataframe at gap points
    # Keep only segments with ≥10 points
```

### Isolated Point Filtering:
```python
def filter_isolated_points(df, neighbor_distance=50, window_size=2, min_neighbors=1):
    # For each point, count neighbors within 50m in ±2 point window
    # Remove points with <1 neighbor
```

---

## 📈 Metrics

| Metric | Value |
|--------|-------|
| **Scripts Created** | 2 |
| **Lines of Code** | ~600 |
| **Functions** | 8 |
| **Data Points Processed** | ~2.5M+ |
| **Routes Segmented** | ~800+ |
| **Processing Time** | ~3-5 min (full dataset) |
| **Storage Output** | ~150 MB (compressed CSVs) |

---

## 🚀 Usage Examples

### Example 1: Process new batch of data
```bash
# Copy new GPS/Stability files to Doback-Data/
cp /path/to/new/*.txt Doback-Data/GPS/
cp /path/to/new/ESTABILIDAD*.txt Doback-Data/Stability/

# Run processing
python Scripts/data-cleaning/process_doback_routes.py

# Check report
cat "Doback-Data/processed data/processing_report.txt"
```

### Example 2: Visualize specific route
```bash
# Find all segments of DOBACK024 from October 5
python Scripts/data-cleaning/visualize_doback_route.py \
  "Doback-Data/processed data/DOBACK024_20251005"

# Map opens automatically in browser
# Shows all segments overlaid with color-coded SI
```

### Example 3: Export to GeoJSON for GIS
```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load processed CSV
df = pd.read_csv("Doback-Data/processed data/DOBACK024_20251005_seg1.csv")

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Export to GeoJSON
gdf.to_file("output/route.geojson", driver="GeoJSON")
```

---

## 🔮 Future Enhancements

Potential improvements for Stage 0 scripts:

1. **Real-time processing**: Watch folders for new files, process automatically
2. **Quality scoring**: Assign quality metrics to each route (GPS fix quality, data completeness)
3. **Outlier detection**: ML-based anomaly detection for SI values
4. **Automated reporting**: Generate HTML reports with maps + statistics
5. **Parallel processing**: Multi-core processing for large batches
6. **Incremental updates**: Only process new/modified files

---

## ✅ Completion Checklist

- [x] Route processing pipeline (`batch_processor.py`)
- [x] Interactive visualization (`route_visualizer.py`)
- [x] Batch processing of all DOBACK devices (023, 024, 027, 028)
- [x] Timestamp matching with configurable tolerance
- [x] Spatial gap detection and segmentation
- [x] Anomaly filtering (GPS jumps, isolated points)
- [x] Multi-segment visualization support
- [x] Automatic segment discovery from base pattern
- [x] Browser auto-open functionality
- [x] Processing report generation
- [x] README documentation
- [x] Integration with project structure

**Status:** ✅ COMPLETED & OPERATIONAL

---

## 📚 References

- GPS data format: `Doback-Data/GPS/GPS_DOBACK*_*.txt`
- Stability data format: `Doback-Data/Stability/ESTABILIDAD_DOBACK*_*.txt`
- Output format: CSV with columns `[timestamp, lat, lon, si, alt, speed_kmh, x_utm, y_utm, roll_deg, pitch_deg, ...]`
- Coordinate system: GPS WGS84 (EPSG:4326) → UTM 30N (EPSG:25830)
- SI scale: [0, 1] where 0 = unstable, 1 = stable

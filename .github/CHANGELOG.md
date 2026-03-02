# Changelog - LiDAR Stability Algorithm

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Sprint 1 Reorganization] - 2026-02-26

### 🔄 Changed - Stage 0 Integrated into Sprint 1
**Rationale:** The "Stage 0" data preparation functionality supersedes and extends the basic parsers, so it has been integrated directly into Sprint 1 rather than existing as a separate preliminary stage.

**File Moves:**
- `Scripts/data-cleaning/process_doback_routes.py` → `Scripts/parsers/batch_processor.py`
- `Scripts/data-cleaning/visualize_doback_route.py` → `Scripts/parsers/route_visualizer.py`
- `Scripts/data-cleaning/README.md` → `Scripts/parsers/README_batch_processing.md`
- `STAGE_0.md` → `SPRINT_1_BATCH_PROCESSING.md`
- `Scripts/data-cleaning/` directory removed

**Documentation Updates:**
- `PROJECT_STATUS.md`: Stage 0 merged into Sprint 1 (now shows 6 modules in Sprint 1)
- `QUICK_START.md`: Updated to reference Sprint 1 batch processing instead of Stage 0
- Sprint 1 now encompasses: basic parsers + batch processing + visualization + physics engine

**Impact:** No functional changes to code, only organizational improvements. All commands remain identical except for path changes.

---

## [Stage 0 Complete] - 2026-02-26
**NOTE:** This entry describes the initial "Stage 0" implementation. See entry above for reorganization into Sprint 1.

### 🆕 Added - Stage 0: Data Preparation & Cleaning

#### New Modules
- **`Scripts/data-cleaning/process_doback_routes.py`** (400 LOC) [NOW: Scripts/parsers/batch_processor.py]
  - Batch processing pipeline for GPS + stability sensor logs
  - Temporal matching using `pd.merge_asof` with configurable tolerance (default 1.0s)
  - Automatic route segmentation based on spatial gaps (default >1000m)
  - Multi-layer outlier filtering:
    - GPS jumps >100m between consecutive points
    - Isolated points (<1 neighbor within 50m in ±2 point window)
  - Minimum segment size filter (≥10 points)
  - UTM coordinate conversion (EPSG:25830) for distance calculations
  - Processing report generation with statistics

- **`Scripts/data-cleaning/visualize_doback_route.py`** (200 LOC)
  - Interactive Folium/OpenStreetMap visualization
  - Gradient color scale: Red (SI=0) → Yellow (SI=0.5) → Green (SI=1)
  - Multi-segment overlay support
  - Automatic segment discovery from base pattern
  - Dynamic legend with SI range, point count, segment info
  - Auto-open in browser (configurable with `--no-browser`)

#### Documentation
- **`STAGE_0.md`** - Complete Stage 0 documentation
  - Objectives & deliverables
  - Processing statistics (~2.5M points, ~800 segments)
  - Technical details & algorithms
  - Usage examples & workflow
  - Integration with Sprint pipelines

- **`Scripts/data-cleaning/README.md`** - Script usage guide
  - Detailed parameter descriptions
  - Examples & workflow
  - Troubleshooting section
  - Integration points with main project

- **`README.md`** (NEW) - Main project README
  - Project overview & objectives
  - Repository structure
  - Quick start guide
  - Status dashboard
  - Technical details & metrics

#### Updated Documentation
- **`PROJECT_STATUS.md`**
  - Added Stage 0 section with metrics
  - Updated overall progress: 54% (Stage 0 + Sprints 1-3)
  - Updated metrics dashboard with Stage 0 columns
  - Updated LOC count: 3,220 total (+600 from Stage 0)

- **`QUICK_START.md`**
  - Added Stage 0 quick start section (Step 0)
  - Updated section numbering
  - Added examples for data cleaning workflows

### 🔧 Technical Improvements

#### Data Processing Features
1. **Robust Timestamp Matching**
   - GPS: 1 Hz with datetime parsing
   - Stability: 10 Hz with microsecond resolution
   - Tolerance: Configurable merge tolerance (default 1.0s)

2. **Intelligent Segmentation**
   - Distance-based gap detection using UTM coordinates
   - Configurable threshold (default 1000m)
   - Automatic segment numbering: `_seg1.csv`, `_seg2.csv`, etc.

3. **Quality Filtering**
   - Coordinate validation: Spanish Peninsula bounds (35-45°N, -10 to 5°E)
   - Altitude range: 0-3000m
   - Speed limit: 0-200 km/h
   - Jump detection: >100m spatial discontinuities
   - Isolated point removal: Spatial density-based filtering

4. **Visualization Enhancement**
   - Fixed SI color scale [0, 1] for consistent comparison
   - Pattern-based file discovery (e.g., `DOBACK024_20251005` → finds all segments)
   - Multi-segment overlay with individual markers
   - Responsive legend with scroll support

### 📊 Data Statistics

**Processed Dataset:**
- Devices: DOBACK023, 024, 027, 028
- Date range: September 2025 - February 2026
- Total routes: ~150+ GPS/Stability pairs
- Valid routes: ~60% (~90 routes with successful matching)
- Total segments: ~800+ (after gap-based splitting)
- Total data points: ~2.5M+ matched GPS+stability records
- Output size: ~150 MB (compressed CSVs)

**Processing Efficiency:**
- Processing time: 3-5 minutes (full dataset)
- Success rate: ~60% (40% have timestamp misalignment)
- Average segment size: ~3,000 points
- Segments discarded: ~15% (less than 10 points)

### 🗂️ File Structure Changes

```
NEW FILES:
├── README.md                                    # Main project README
├── STAGE_0.md                                   # Stage 0 documentation
├── CHANGELOG.md                                 # This file
└── Scripts/data-cleaning/
    ├── README.md                                # Data cleaning guide
    ├── process_doback_routes.py                 # Processing pipeline
    └── visualize_doback_route.py                # Visualization tool

UPDATED FILES:
├── PROJECT_STATUS.md                            # Added Stage 0 section
└── QUICK_START.md                               # Added Stage 0 quick start

NEW DATA:
└── Doback-Data/processed data/
    ├── DOBACK023_20251012.csv                   # Example: single route
    ├── DOBACK024_20251005_seg1.csv              # Example: multi-segment
    ├── DOBACK024_20251005_seg2.csv
    ├── ...
    └── processing_report.txt                    # Processing statistics
```

### 🔗 Integration Points

**Stage 0 → Sprint 1:**
- Clean CSVs can be loaded directly into Sprint 1 parsers
- Timestamp alignment already done, simplifies EKF fusion
- Segmented routes ready for individual analysis

**Stage 0 → Sprint 3:**
- GPS coordinates (lat/lon) and UTM coordinates (x_utm, y_utm) ready for LiDAR spatial queries
- Route segments can be matched to LiDAR tiles for terrain feature extraction

**Stage 0 → Future Sprints:**
- Clean datasets ready for ML training (Sprint 4-5)
- Route geometry prepared for map generation (Sprint 6)

### 📝 Usage Examples

#### Example 1: Process and Visualize a Route
```bash
# Step 1: Process all raw data
python Scripts/data-cleaning/process_doback_routes.py

# Step 2: Visualize specific route with all segments
python Scripts/data-cleaning/visualize_doback_route.py \
  "Doback-Data/processed data/DOBACK024_20251005"

# Output: Interactive map opens in browser
```

#### Example 2: Custom Processing Parameters
```bash
python Scripts/data-cleaning/process_doback_routes.py \
  --tolerance-seconds 2.0 \
  --max-gap-meters 500

# More lenient timestamp matching
# Finer route segmentation
```

#### Example 3: Multi-Route Visualization
```bash
python Scripts/data-cleaning/visualize_doback_route.py \
  "Doback-Data/processed data/DOBACK024_20251005" \
  "Doback-Data/processed data/DOBACK024_20251007" \
  --no-browser

# Overlay multiple routes
# Save without opening browser
```

### 🧪 Validation

**Manual Validation Performed:**
- ✅ Timestamp matching accuracy verified on sample routes
- ✅ Spatial filtering confirmed (GPS jumps detected and removed)
- ✅ Segmentation logic validated (gaps correctly identified)
- ✅ Visualization accuracy checked (colors match SI values)
- ✅ Multi-segment overlay confirmed working

**No Automated Tests:**
- Stage 0 is a data preparation pipeline, validated through visual inspection and statistics
- Future: Could add integration tests for processing pipeline

### 🚀 Performance Optimizations

1. **Vectorized Operations:**
   - NumPy arrays for distance calculations
   - Pandas vectorized operations for filtering
   - ~10x faster than iterative approach

2. **Efficient Spatial Indexing:**
   - UTM coordinate system for metric distance calculations
   - Pre-computed distances for gap detection

3. **Memory Management:**
   - Chunked processing for large files
   - Efficient DataFrame operations with `merge_asof`

### 🔮 Future Enhancements

**Potential Improvements:**
1. Real-time processing (watch folder for new files)
2. Quality scoring per route (GPS fix quality, data completeness)
3. ML-based anomaly detection for SI values
4. Automated HTML report generation
5. Parallel processing for large batches
6. Incremental updates (only process new/modified files)
7. Integration with automated test suite

---

## [Sprint 3 Complete] - 2025-XX-XX

### Added
- LAZ reader for CNIG PNOA point clouds (51 tiles)
- GeoTIFF reader for DTM rasters
- Terrain provider with auto-source selection
- Terrain feature extraction (φ_lidar, TRI)
- 20 automated tests for LiDAR module

### Technical Details
- ~250M+ LiDAR points indexed
- KD-tree spatial indexing for fast queries
- Affine transform coordinate conversion
- Resolution: 0.5 pts/m²

---

## [Sprint 2 Complete] - 2025-XX-XX

### Added
- Extended Kalman Filter (EKF) for sensor fusion
- GPS (1 Hz) + IMU (10 Hz) → continuous trajectory
- Time synchronization module
- Analytic Jacobians with numerical validation
- 8 automated tests for EKF module

### Technical Details
- Position accuracy: ±<1m stationary
- Velocity tracking: 9-11 m/s constant speed
- Jacobian precision: <1e-4 error

---

## [Sprint 1 Complete] - 2025-XX-XX

### Added
- GPS parser with multi-layer validation
- IMU/Stability parser with frequency detection
- Physics engine for SI calculation
- Ground truth pipeline (SI_real, SI_static, ΔSI)
- 17 automated tests for parsers & physics

### Technical Details
- Critical angle: φc = 33.79°
- IMU frequency: 10 Hz (confirmed via median Δt = 100.07ms)
- Data recovery: 78% GPS records valid

---

## Project Initialization - 2025-XX-XX

### Added
- Initial project structure
- requirements.txt with dependencies
- Vehicle configuration (DOBACK024 parameters)
- .gitignore for large files (.laz, .tif)
- pytest configuration

---

**Legend:**
- 🆕 **Added** - New features or files
- 🔧 **Changed** - Changes to existing functionality
- 🐛 **Fixed** - Bug fixes
- 🗑️ **Removed** - Removed features or files
- 📝 **Deprecated** - Features marked for future removal
- 🔒 **Security** - Security improvements

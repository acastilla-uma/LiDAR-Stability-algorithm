# Changelog

All notable changes to this project are documented in this file.

## 2026-03-20 - Reorganization Hardening, Documentation Refresh, and Performance Improvements

- Updated repository path resolution across core CLI scripts after migration to `src/lidar_stability`.
- Standardized defaults and references to `Doback-Data/processed-data` (hyphenated folder name).
- Fixed package import bootstrapping (`sys.path`) for standalone script execution in the new layout.
- Fixed `audit_pipeline_coverage.py` root detection and added explicit message when no raw pairs are found.
- Updated physics comparison default config path to `src/lidar_stability/config/vehicle.yaml`.
- Optimized terrain feature extraction hotspots:
	- Avoided unnecessary full slope computation in `compute_phi_lidar` when cross-profile is valid.
	- Vectorized ruggedness calculation.
	- Reduced repeated overhead in route feature enrichment (tile index/query cache + array-based dataframe assignment).
- Refreshed project documentation:
	- Updated `README.md`, `docs/QUICK_START.md`, `docs/PROJECT_STATUS.md`, `docs/ROADMAP.md`.
	- Updated technical guides and added missing CLI reference documentation.
- Validated key CLIs with smoke checks and `--help` entrypoint verification.

## 2026-02-26 - Sprint 1 Reorganization and Batch Processing Consolidation
- Integrated Stage 0 work into Sprint 1 to simplify delivery flow.
- Moved data-cleaning responsibilities into parser-focused production modules.
- Standardized operational entry points around batch processing and route visualization.
- Consolidated project structure to improve maintainability and sprint traceability.

## 2026-02-26 - Sprint 1 Delivery Completion
- Completed production batch processor for GPS/stability matching, anomaly filtering, and route segmentation.
- Added route visualization tooling with SI color mapping and multi-segment support.
- Finalized Sprint 1 test coverage and validation workflow.

## 2026-Q1 - Sprint 3 LiDAR and Terrain Features Completion
- Delivered LAZ and GeoTIFF readers for terrain ingestion.
- Added unified terrain provider with source fallback strategy.
- Implemented terrain feature extraction (`phi_lidar`, `TRI`, ruggedness, elevation stats).
- Completed Sprint 3 test suite and integration notes for downstream pipeline stages.

## 2026-Q1 - Sprint 3 Consolidated Summary
- Confirmed Sprint 3 modules as production-ready for terrain enrichment.
- Documented integration boundaries between map-matching, terrain features, and modeling stages.
- Captured operational readiness for Sprint 4 model development.

# Changelog

All notable changes to this project are documented in this file.

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

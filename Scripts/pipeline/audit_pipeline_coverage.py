#!/usr/bin/env python3
"""
Audit processing coverage across the route pipeline.

Measures how many raw DOBACK routes are:
1) available as raw GPS+Stability pairs,
2) processed into CSV,
3) map-matched,
4) enriched with terrain features (featured).

It also reports totals at file/segment level and can export a detailed CSV.

Usage:
    python Scripts/pipeline/audit_pipeline_coverage.py
    python Scripts/pipeline/audit_pipeline_coverage.py --output output/results/pipeline_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = [
    "phi_lidar",
    "phi_lidar_deg",
    "tri",
    "ruggedness",
    "z_min",
    "z_max",
    "z_mean",
    "z_std",
    "z_range",
    "n_points_used",
]


def normalize_cli_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    normalized = path_value.replace("\\", "/")
    return str(Path(normalized))


@dataclass
class CsvInfo:
    path: Path
    row_count: int
    has_features_columns: bool
    has_features_values: bool


def key_from_name(name: str, prefix: str) -> str | None:
    if "realtime" in name.lower():
        return None
    m = re.match(rf"{prefix}_(DOBACK\d+?)_(.+)\.txt$", name)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def discover_raw_pairs(data_dir: Path) -> dict[str, dict[str, Path]]:
    gps_dir = data_dir / "GPS"
    stab_dir = data_dir / "Stability"

    gps_map: dict[str, Path] = {}
    stab_map: dict[str, Path] = {}

    for path in sorted(gps_dir.glob("GPS_DOBACK*_*.txt")):
        key = key_from_name(path.name, "GPS")
        if key:
            gps_map[key] = path

    for path in sorted(stab_dir.glob("ESTABILIDAD_DOBACK*_*.txt")):
        key = key_from_name(path.name, "ESTABILIDAD")
        if key:
            stab_map[key] = path

    common_keys = sorted(set(gps_map) & set(stab_map))
    return {
        key: {"gps": gps_map[key], "stability": stab_map[key]}
        for key in common_keys
    }


def group_csvs_by_base(directory: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("."):
            continue
        stem = path.stem
        base = re.sub(r"_seg\d+$", "", stem)
        grouped.setdefault(base, []).append(path)
    return grouped


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def inspect_csv(path: Path) -> CsvInfo:
    row_count = 0
    has_features_columns = False
    has_features_values = False

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_features_columns = all(col in fieldnames for col in FEATURE_COLUMNS)

        for row in reader:
            row_count += 1
            if has_features_columns and not has_features_values:
                n_points = row.get("n_points_used")
                if _is_number(n_points) and float(n_points) > 0:
                    has_features_values = True
                    continue

                for col in FEATURE_COLUMNS:
                    if col == "n_points_used":
                        continue
                    value = row.get(col)
                    if value not in (None, "", "nan", "NaN"):
                        has_features_values = True
                        break

    return CsvInfo(
        path=path,
        row_count=row_count,
        has_features_columns=has_features_columns,
        has_features_values=has_features_values,
    )


def inspect_many(paths: Iterable[Path]) -> list[CsvInfo]:
    return [inspect_csv(path) for path in paths]


def build_rows(
    raw_pairs: dict[str, dict[str, Path]],
    processed_dir: Path,
    mapmatched_dir: Path,
    featured_dir: Path,
) -> list[dict]:
    processed_grouped = group_csvs_by_base(processed_dir)
    mapmatched_grouped = group_csvs_by_base(mapmatched_dir)
    featured_grouped = group_csvs_by_base(featured_dir)

    rows: list[dict] = []
    for key, raw_paths in raw_pairs.items():
        processed_files = processed_grouped.get(key, [])
        mapmatched_files = mapmatched_grouped.get(key, [])
        featured_files = featured_grouped.get(key, [])

        processed_info = inspect_many(processed_files) if processed_files else []
        mapmatched_info = inspect_many(mapmatched_files) if mapmatched_files else []
        featured_info = inspect_many(featured_files) if featured_files else []

        rows.append(
            {
                "raw_key": key,
                "gps_raw": raw_paths["gps"].name,
                "stability_raw": raw_paths["stability"].name,
                "processed_files": len(processed_info),
                "processed_rows": sum(info.row_count for info in processed_info),
                "mapmatched_files": len(mapmatched_info),
                "mapmatched_rows": sum(info.row_count for info in mapmatched_info),
                "feature_files": sum(1 for info in featured_info if info.has_features_columns),
                "feature_value_files": sum(1 for info in featured_info if info.has_features_values),
                "feature_rows": sum(info.row_count for info in featured_info if info.has_features_columns),
                "feature_value_rows": sum(info.row_count for info in featured_info if info.has_features_values),
                "is_processed": int(len(processed_info) > 0),
                "is_mapmatched": int(len(mapmatched_info) > 0),
                "has_features": int(any(info.has_features_columns for info in featured_info)),
                "has_feature_values": int(any(info.has_features_values for info in featured_info)),
                "processed_file_names": ";".join(info.path.name for info in processed_info),
                "mapmatched_file_names": ";".join(info.path.name for info in mapmatched_info),
                "feature_file_names": ";".join(info.path.name for info in featured_info if info.has_features_columns),
                "feature_value_file_names": ";".join(info.path.name for info in featured_info if info.has_features_values),
            }
        )

    return rows


def summarize(rows: list[dict]) -> dict:
    total_raw = len(rows)
    processed_routes = sum(row["is_processed"] for row in rows)
    mapmatched_routes = sum(row["is_mapmatched"] for row in rows)
    featured_routes = sum(row["has_features"] for row in rows)
    featured_value_routes = sum(row["has_feature_values"] for row in rows)

    return {
        "raw_pairs_total": total_raw,
        "raw_pairs_processed": processed_routes,
        "raw_pairs_mapmatched": mapmatched_routes,
        "raw_pairs_with_features": featured_routes,
        "raw_pairs_with_feature_values": featured_value_routes,
        "processed_files_total": sum(row["processed_files"] for row in rows),
        "mapmatched_files_total": sum(row["mapmatched_files"] for row in rows),
        "feature_files_total": sum(row["feature_files"] for row in rows),
        "feature_value_files_total": sum(row["feature_value_files"] for row in rows),
        "processed_rows_total": sum(row["processed_rows"] for row in rows),
        "mapmatched_rows_total": sum(row["mapmatched_rows"] for row in rows),
        "feature_rows_total": sum(row["feature_rows"] for row in rows),
        "feature_value_rows_total": sum(row["feature_value_rows"] for row in rows),
        "processed_pct": (processed_routes / total_raw * 100.0) if total_raw else 0.0,
        "mapmatched_pct": (mapmatched_routes / total_raw * 100.0) if total_raw else 0.0,
        "features_pct": (featured_routes / total_raw * 100.0) if total_raw else 0.0,
        "feature_values_pct": (featured_value_routes / total_raw * 100.0) if total_raw else 0.0,
    }


def summarize_by_device(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        device = row["raw_key"].split("_", 1)[0]
        grouped[device].append(row)

    summaries: list[dict] = []
    for device in sorted(grouped):
        items = grouped[device]
        total = len(items)
        processed = sum(item["is_processed"] for item in items)
        mapmatched = sum(item["is_mapmatched"] for item in items)
        features = sum(item["has_features"] for item in items)
        feature_values = sum(item["has_feature_values"] for item in items)
        summaries.append(
            {
                "device": device,
                "raw_pairs": total,
                "processed_routes": processed,
                "mapmatched_routes": mapmatched,
                "feature_routes": features,
                "feature_value_routes": feature_values,
                "processed_pct": (processed / total * 100.0) if total else 0.0,
                "mapmatched_pct": (mapmatched / total * 100.0) if total else 0.0,
                "features_pct": (features / total * 100.0) if total else 0.0,
                "feature_values_pct": (feature_values / total * 100.0) if total else 0.0,
            }
        )
    return summaries


def print_summary(summary: dict) -> None:
    print("\n=== Pipeline coverage audit ===")
    print(f"Raw GPS+Stability pairs:   {summary['raw_pairs_total']}")
    print(f"Processed routes:          {summary['raw_pairs_processed']} ({summary['processed_pct']:.1f}%)")
    print(f"Map-matched routes:        {summary['raw_pairs_mapmatched']} ({summary['mapmatched_pct']:.1f}%)")
    print(f"Routes with features:      {summary['raw_pairs_with_features']} ({summary['features_pct']:.1f}%)")
    print(f"Routes with feature values:{summary['raw_pairs_with_feature_values']} ({summary['feature_values_pct']:.1f}%)")
    print()
    print(f"Processed CSV files:       {summary['processed_files_total']}")
    print(f"Map-matched CSV files:     {summary['mapmatched_files_total']}")
    print(f"Featured CSV files:        {summary['feature_files_total']}")
    print(f"Feature-value CSV files:   {summary['feature_value_files_total']}")
    print()
    print(f"Processed rows total:      {summary['processed_rows_total']:,}")
    print(f"Map-matched rows total:    {summary['mapmatched_rows_total']:,}")
    print(f"Feature rows total:        {summary['feature_rows_total']:,}")
    print(f"Feature-value rows total:  {summary['feature_value_rows_total']:,}")


def print_simple_summary(device_summaries: list[dict]) -> None:
    print("\n=== Resumen simplificado por DOBACK ===")
    for item in device_summaries:
        print(
            f"{item['device']}: raw={item['raw_pairs']}, "
            f"processed={item['processed_pct']:.1f}%, "
            f"map-matched={item['mapmatched_pct']:.1f}%, "
            f"features={item['features_pct']:.1f}%, "
            f"feature-values={item['feature_values_pct']:.1f}%"
        )


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["raw_key"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit how many raw DOBACK routes are processed, map-matched, and enriched with terrain features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(BASE / "Doback-Data"))
    parser.add_argument("--processed-dir", default=str(BASE / "Doback-Data" / "processed-data"))
    parser.add_argument("--mapmatched-dir", default=str(BASE / "Doback-Data" / "map-matched"))
    parser.add_argument("--featured-dir", default=str(BASE / "Doback-Data" / "featured"))
    parser.add_argument("--output", default=str(BASE / "output" / "pipeline_audit.csv"))
    parser.add_argument("--json", action="store_true", help="Also print summary as JSON")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Mostrar solo porcentajes por dispositivo y no guardar CSV",
    )
    args = parser.parse_args()

    args.data_dir = normalize_cli_path(args.data_dir)
    args.processed_dir = normalize_cli_path(args.processed_dir)
    args.mapmatched_dir = normalize_cli_path(args.mapmatched_dir)
    args.featured_dir = normalize_cli_path(args.featured_dir)
    args.output = normalize_cli_path(args.output)

    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    mapmatched_dir = Path(args.mapmatched_dir)
    featured_dir = Path(args.featured_dir)
    output_path = Path(args.output)

    raw_pairs = discover_raw_pairs(data_dir)
    rows = build_rows(raw_pairs, processed_dir, mapmatched_dir, featured_dir)
    summary = summarize(rows)
    device_summaries = summarize_by_device(rows)

    if args.simple:
        print_simple_summary(device_summaries)
        if args.json:
            print("\n" + json.dumps(device_summaries, indent=2, ensure_ascii=False))
        return 0

    print_summary(summary)
    write_csv(rows, output_path)
    print(f"\nDetailed CSV: {output_path}")

    if args.json:
        print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    incomplete = [row for row in rows if not row["has_features"]]
    if incomplete:
        print("\nPrimeras rutas sin features:")
        for row in incomplete[:10]:
            print(
                f"- {row['raw_key']}: processed={row['processed_files']}, "
                f"mapmatched={row['mapmatched_files']}, feature_files={row['feature_value_files']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate an end-to-end EDA notebook for the LiDAR stability pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NOTEBOOK_INTRO_MD = """# EDA del pipeline: crudo -> procesado -> map-matched -> featured

Este notebook analiza la evolucion de los datos a lo largo del pipeline:

1. Comparacion de perdida de datos antes y despues del parseo (estabilidad + GPS vs processed-data).
2. Distribuciones por variable, porcentaje de invalidos y outliers por etapa.
3. Analisis de calidad del map-matching con metricas de distancia.
4. Perdida de datos y distribuciones tras feature extraction.

Variables base analizadas: `ax`, `gx`, `gz`, `gy`, `ay`, `az`, `roll`, `pitch`, `yaw`, `si`, `accmag`, `velocidad (km/h)`.
"""


NOTEBOOK_SETUP_CODE = """from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

REPO_ROOT = Path.cwd()
DATA_ROOT = REPO_ROOT / "Doback-Data"
GPS_DIR = DATA_ROOT / "GPS"
STABILITY_DIR = DATA_ROOT / "Stability"
PROCESSED_DIR = DATA_ROOT / "processed-data"
MAPMATCHED_DIR = DATA_ROOT / "map-matched"
FEATURED_DIR = DATA_ROOT / "featured"

MAX_PLOT_ROWS = 250000  # limita coste de render en datasets grandes
RANDOM_SEED = 42

print(f"Repo root: {REPO_ROOT}")
for p in [GPS_DIR, STABILITY_DIR, PROCESSED_DIR, MAPMATCHED_DIR, FEATURED_DIR]:
    print(f"- {p}: {'OK' if p.exists() else 'MISSING'}")
"""


NOTEBOOK_HELPERS_CODE = """# Variables objetivo y aliases por posibles nombres en distintas etapas
BASE_VARIABLES = {
    "ax": ["ax"],
    "gx": ["gx"],
    "gz": ["gz"],
    "gy": ["gy"],
    "ay": ["ay"],
    "az": ["az"],
    "roll": ["roll"],
    "pitch": ["pitch"],
    "yaw": ["yaw"],
    "si": ["si"],
    "accmag": ["accmag"],
    "velocidad_kmh": ["speed_kmh", "velocidad_kmh", "Velocidad(km/h)", "Velocida", "velocidad"],
}

FEATURE_VARIABLES = {
    "phi_lidar": ["phi_lidar"],
    "phi_lidar_deg": ["phi_lidar_deg"],
    "tri": ["tri"],
    "ruggedness": ["ruggedness"],
    "z_min": ["z_min"],
    "z_max": ["z_max"],
    "z_mean": ["z_mean"],
    "z_std": ["z_std"],
    "z_range": ["z_range"],
    "n_points_used": ["n_points_used"],
}


def file_key_from_name(name: str) -> str | None:
    # Ejemplos esperados:
    # - GPS_DOBACK024_20251230.txt
    # - ESTABILIDAD_DOBACK024_20251230.txt
    # - DOBACK024_20251230.csv
    # - DOBACK024_20251230_seg1.csv
    m = re.search(r"(DOBACK\d+_\d{8})", name, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()


def count_data_rows_txt(path: Path) -> int:
    # Los ficheros crudos tienen cabeceras en primeras lineas; contamos solo lineas con separadores.
    rows = 0
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if ";" in line:
                rows += 1
    # Ajuste conservador para excluir lineas de cabecera tipicas
    return max(0, rows - 2)


def count_rows_csv(path: Path) -> int:
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:
        return 0


def list_csv_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.csv")) if folder.exists() else []


def list_txt_files(folder: Path, prefix: str) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{prefix}*.txt"))


def compare_stage_counts(before_counts: dict[str, int], after_counts: dict[str, int], stage_name: str) -> pd.DataFrame:
    keys = sorted(set(before_counts) | set(after_counts))
    rows = []
    for key in keys:
        before = int(before_counts.get(key, 0))
        after = int(after_counts.get(key, 0))
        lost = max(0, before - after)
        loss_pct = (100.0 * lost / before) if before > 0 else np.nan
        rows.append({
            "stage": stage_name,
            "key": key,
            "before_rows": before,
            "after_rows": after,
            "lost_rows": lost,
            "loss_pct": loss_pct,
        })
    return pd.DataFrame(rows)


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in aliases:
        match = lower_map.get(c.lower())
        if match:
            return match
    return None


def _iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def analyze_variable(df: pd.DataFrame, var_name: str, aliases: list[str], stage_label: str) -> dict:
    col = _resolve_column(df, aliases)
    if col is None:
        return {
            "stage": stage_label,
            "variable": var_name,
            "column_used": None,
            "rows": len(df),
            "valid_rows": 0,
            "invalid_pct": 100.0 if len(df) else np.nan,
            "outlier_pct_over_valid": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    series = pd.to_numeric(df[col], errors="coerce")
    total = len(series)
    valid_mask = series.notna() & np.isfinite(series)
    valid = series[valid_mask]

    invalid_pct = (100.0 * (total - valid_mask.sum()) / total) if total > 0 else np.nan
    if len(valid) > 0:
        out_mask = _iqr_outlier_mask(valid)
        outlier_pct = 100.0 * out_mask.mean()
    else:
        outlier_pct = np.nan

    return {
        "stage": stage_label,
        "variable": var_name,
        "column_used": col,
        "rows": total,
        "valid_rows": int(valid_mask.sum()),
        "invalid_pct": invalid_pct,
        "outlier_pct_over_valid": outlier_pct,
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "std": float(valid.std()) if len(valid) else np.nan,
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
    }


def plot_variable_distribution(df: pd.DataFrame, var_name: str, aliases: list[str], stage_label: str) -> dict:
    summary = analyze_variable(df, var_name, aliases, stage_label)
    col = summary["column_used"]

    if col is None:
        print(f"[{stage_label}] {var_name}: columna no encontrada. Aliases: {aliases}")
        return summary

    series = pd.to_numeric(df[col], errors="coerce")
    valid = series[series.notna() & np.isfinite(series)]

    if len(valid) == 0:
        print(f"[{stage_label}] {var_name}: sin valores numericos validos")
        return summary

    if len(valid) > MAX_PLOT_ROWS:
        plot_vals = valid.sample(MAX_PLOT_ROWS, random_state=RANDOM_SEED)
    else:
        plot_vals = valid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(plot_vals, bins=60, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"{stage_label} - {var_name} (histograma)")
    axes[0].set_xlabel(col)

    sns.boxplot(x=plot_vals, ax=axes[1], color="#ff7f0e")
    axes[1].set_title(f"{stage_label} - {var_name} (boxplot)")
    axes[1].set_xlabel(col)

    fig.tight_layout()
    plt.show()

    print(
        f"[{stage_label}] {var_name} | col={col} | "
        f"invalid_pct={summary['invalid_pct']:.2f}% | "
        f"outlier_pct={summary['outlier_pct_over_valid']:.2f}% | "
        f"valid_rows={summary['valid_rows']}"
    )
    return summary


def plot_variable_comparison_same_scale(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    var_name: str,
    aliases: list[str],
    left_label: str,
    right_label: str,
) -> tuple[dict, dict]:
    left_summary = analyze_variable(df_left, var_name, aliases, left_label)
    right_summary = analyze_variable(df_right, var_name, aliases, right_label)

    left_col = left_summary["column_used"]
    right_col = right_summary["column_used"]

    if left_col is None or right_col is None:
        print(
            f"[{var_name}] comparacion omitida: columna no encontrada en uno de los datasets "
            f"({left_label}={left_col}, {right_label}={right_col})"
        )
        return left_summary, right_summary

    left_series = pd.to_numeric(df_left[left_col], errors="coerce")
    right_series = pd.to_numeric(df_right[right_col], errors="coerce")
    left_valid = left_series[left_series.notna() & np.isfinite(left_series)]
    right_valid = right_series[right_series.notna() & np.isfinite(right_series)]

    if len(left_valid) == 0 or len(right_valid) == 0:
        print(
            f"[{var_name}] comparacion omitida: faltan valores validos "
            f"({left_label}={len(left_valid)}, {right_label}={len(right_valid)})"
        )
        return left_summary, right_summary

    left_plot = (
        left_valid.sample(MAX_PLOT_ROWS, random_state=RANDOM_SEED)
        if len(left_valid) > MAX_PLOT_ROWS
        else left_valid
    )
    right_plot = (
        right_valid.sample(MAX_PLOT_ROWS, random_state=RANDOM_SEED)
        if len(right_valid) > MAX_PLOT_ROWS
        else right_valid
    )

    combined = pd.concat([left_plot, right_plot], ignore_index=True)
    x_min = float(combined.min())
    x_max = float(combined.max())
    if x_min == x_max:
        delta = 1.0 if x_min == 0 else abs(x_min) * 0.01
        x_min -= delta
        x_max += delta
    bins = np.linspace(x_min, x_max, 61)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

    sns.histplot(left_plot, bins=bins, kde=True, ax=axes[0, 0], color="#1f77b4")
    axes[0, 0].set_title(f"{left_label} - {var_name} (histograma)")
    axes[0, 0].set_xlim(x_min, x_max)

    sns.histplot(right_plot, bins=bins, kde=True, ax=axes[0, 1], color="#2ca02c")
    axes[0, 1].set_title(f"{right_label} - {var_name} (histograma)")
    axes[0, 1].set_xlim(x_min, x_max)

    sns.boxplot(x=left_plot, ax=axes[1, 0], color="#ff7f0e")
    axes[1, 0].set_title(f"{left_label} - {var_name} (boxplot)")
    axes[1, 0].set_xlim(x_min, x_max)

    sns.boxplot(x=right_plot, ax=axes[1, 1], color="#d62728")
    axes[1, 1].set_title(f"{right_label} - {var_name} (boxplot)")
    axes[1, 1].set_xlim(x_min, x_max)

    # Igualamos escala Y de histogramas para comparacion visual justa.
    y_max = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(0, y_max)
    axes[0, 1].set_ylim(0, y_max)

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Comparacion con misma escala: {var_name}", y=1.02)
    fig.tight_layout()
    plt.show()

    print(
        f"[{var_name}] escalas igualadas | x_range=({x_min:.4f}, {x_max:.4f}) | "
        f"y_hist_max={y_max:.2f}"
    )
    print(
        f"[{left_label}] invalid_pct={left_summary['invalid_pct']:.2f}% "
        f"outlier_pct={left_summary['outlier_pct_over_valid']:.2f}%"
    )
    print(
        f"[{right_label}] invalid_pct={right_summary['invalid_pct']:.2f}% "
        f"outlier_pct={right_summary['outlier_pct_over_valid']:.2f}%"
    )

    return left_summary, right_summary


def load_stability_raw_dataframe(stability_files: list[Path]) -> pd.DataFrame:
    rows = []
    expected_cols = [
        "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw",
        "timeantwifi", "usciclo1", "usciclo2", "usciclo3", "usciclo4", "usciclo5",
        "si", "accmag", "microsds", "k3"
    ]
    for fp in stability_files:
        key = file_key_from_name(fp.name)
        with fp.open("r", encoding="latin-1", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        data_lines = []
        for ln in lines:
            # En bruto las filas de datos suelen ser valores separados por ';'.
            if ln.count(";") >= 10 and not ln.upper().startswith("ESTABILIDAD") and not ln.lower().startswith("ax;"):
                data_lines.append(ln)
        for ln in data_lines:
            parts = [p.strip() for p in ln.split(";")]
            if len(parts) < len(expected_cols):
                continue
            row = {col: parts[i] for i, col in enumerate(expected_cols)}
            row["source_key"] = key
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=expected_cols + ["source_key"])

    df = pd.DataFrame(rows)
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_stage_dataframe(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    for fp in csv_files:
        try:
            dfi = pd.read_csv(fp)
        except Exception:
            continue
        dfi["source_file"] = fp.name
        dfi["source_key"] = file_key_from_name(fp.name)
        frames.append(dfi)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def print_global_loss(df_cmp: pd.DataFrame, label: str) -> None:
    before_total = int(df_cmp["before_rows"].sum())
    after_total = int(df_cmp["after_rows"].sum())
    lost_total = max(0, before_total - after_total)
    loss_pct = (100.0 * lost_total / before_total) if before_total > 0 else np.nan
    print(f"{label}: before={before_total:,} after={after_total:,} lost={lost_total:,} loss_pct={loss_pct:.2f}%")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))
"""


NOTEBOOK_PHASE2_MD = """## Fase 2 - Crudo vs procesado (comparacion antes/despues del parse)

En esta fase no se implementa parseo nuevo: se comparan directamente los archivos crudos y los procesados para medir la perdida de filas.
Tambien se generan graficas de distribucion por variable con porcentaje de invalidos y outliers.
"""


NOTEBOOK_PHASE2_CODE = """# 1) Conteos por etapa cruda y procesada

gps_files = list_txt_files(GPS_DIR, "GPS_")
stability_files = list_txt_files(STABILITY_DIR, "ESTABILIDAD_")
processed_csv = list_csv_files(PROCESSED_DIR)

print(f"GPS files: {len(gps_files)}")
print(f"Stability files: {len(stability_files)}")
print(f"Processed csv files: {len(processed_csv)}")

raw_stability_counts = {}
for fp in stability_files:
    key = file_key_from_name(fp.name)
    if key is None:
        continue
    raw_stability_counts[key] = raw_stability_counts.get(key, 0) + count_data_rows_txt(fp)

processed_counts = {}
for fp in processed_csv:
    key = file_key_from_name(fp.name)
    if key is None:
        continue
    processed_counts[key] = processed_counts.get(key, 0) + count_rows_csv(fp)

df_parse_cmp = compare_stage_counts(raw_stability_counts, processed_counts, "raw_stability_vs_processed")
print_global_loss(df_parse_cmp, "Perdida parse (stability -> processed)")

display(df_parse_cmp.sort_values("loss_pct", ascending=False).head(20))

# 2) Dataframes para distribuciones
raw_stability_df = load_stability_raw_dataframe(stability_files)
processed_df = load_stage_dataframe(processed_csv)

print(f"raw_stability_df shape: {raw_stability_df.shape}")
print(f"processed_df shape: {processed_df.shape}")
"""


NOTEBOOK_PHASE2_PLOTS_CODE = """# Graficas de distribucion por variable (fase 2)

raw_summaries = []
processed_summaries = []

for var_name, aliases in BASE_VARIABLES.items():
    print("=" * 90)
    print(f"Variable: {var_name} (RAW STABILITY vs PROCESSED, misma escala)")
    raw_sum, processed_sum = plot_variable_comparison_same_scale(
        raw_stability_df,
        processed_df,
        var_name,
        aliases,
        "raw_stability",
        "processed",
    )
    raw_summaries.append(raw_sum)
    processed_summaries.append(processed_sum)

raw_summary_df = pd.DataFrame(raw_summaries)
processed_summary_df = pd.DataFrame(processed_summaries)

print("\\nResumen RAW")
display(raw_summary_df)
print("\\nResumen PROCESSED")
display(processed_summary_df)
"""


NOTEBOOK_PHASE3_MD = """## Fase 3 - Evaluacion de map-matching

Se calcula:
- Porcentaje de filas perdidas de `processed-data` a `map-matched`.
- Calidad del ajuste por distancia entre coordenada original y coordenada ajustada.
- Distribuciones por variable en la salida map-matched.
"""


NOTEBOOK_PHASE3_CODE = """mapmatched_csv = list_csv_files(MAPMATCHED_DIR)
mapmatched_df = load_stage_dataframe(mapmatched_csv)

print(f"mapmatched_df shape: {mapmatched_df.shape}")

mapmatched_counts = {}
for fp in mapmatched_csv:
    key = file_key_from_name(fp.name)
    if key is None:
        continue
    mapmatched_counts[key] = mapmatched_counts.get(key, 0) + count_rows_csv(fp)

df_mm_cmp = compare_stage_counts(processed_counts, mapmatched_counts, "processed_vs_mapmatched")
print_global_loss(df_mm_cmp, "Perdida procesado -> map-matched")
display(df_mm_cmp.sort_values("loss_pct", ascending=False).head(20))

# Metricas de distancia del map-matching
mm_dist_series = pd.Series(dtype=float)
if not mapmatched_df.empty:
    if all(c in mapmatched_df.columns for c in ["lat_raw", "lon_raw", "lat", "lon"]):
        lat1 = pd.to_numeric(mapmatched_df["lat_raw"], errors="coerce")
        lon1 = pd.to_numeric(mapmatched_df["lon_raw"], errors="coerce")
        lat2 = pd.to_numeric(mapmatched_df["lat"], errors="coerce")
        lon2 = pd.to_numeric(mapmatched_df["lon"], errors="coerce")
        ok = lat1.notna() & lon1.notna() & lat2.notna() & lon2.notna()
        mm_dist_series = pd.Series(haversine_m(lat1[ok], lon1[ok], lat2[ok], lon2[ok]))
    elif "dist_to_road_m" in mapmatched_df.columns:
        mm_dist_series = pd.to_numeric(mapmatched_df["dist_to_road_m"], errors="coerce").dropna()

if len(mm_dist_series) > 0:
    print("Metricas de distancia map-matching (m)")
    print(mm_dist_series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    plt.figure(figsize=(10, 4))
    sns.histplot(mm_dist_series, bins=60, kde=True, color="#2ca02c")
    plt.title("Distribucion de distancia del map-matching (m)")
    plt.xlabel("distancia (m)")
    plt.tight_layout()
    plt.show()
else:
    print("No hay columnas suficientes para calcular metrica de distancia de map-matching")
"""


NOTEBOOK_PHASE3_PLOTS_CODE = """mapmatched_summaries = []
for var_name, aliases in BASE_VARIABLES.items():
    print("=" * 90)
    print(f"Variable: {var_name} (MAP-MATCHED)")
    mapmatched_summaries.append(plot_variable_distribution(mapmatched_df, var_name, aliases, "map_matched"))

mapmatched_summary_df = pd.DataFrame(mapmatched_summaries)
print("\\nResumen MAP-MATCHED")
display(mapmatched_summary_df)
"""


NOTEBOOK_PHASE4_MD = """## Fase 4 - Evaluacion de feature extraction

Se calcula:
- Porcentaje de filas perdidas de `map-matched` a `featured`.
- Distribuciones por variable base + variables nuevas calculadas en feature extraction.
- Porcentaje de invalidos y outliers por variable.
"""


NOTEBOOK_PHASE4_CODE = """featured_csv = list_csv_files(FEATURED_DIR)
featured_df = load_stage_dataframe(featured_csv)

print(f"featured_df shape: {featured_df.shape}")

featured_counts = {}
for fp in featured_csv:
    key = file_key_from_name(fp.name)
    if key is None:
        continue
    featured_counts[key] = featured_counts.get(key, 0) + count_rows_csv(fp)

df_feat_cmp = compare_stage_counts(mapmatched_counts, featured_counts, "mapmatched_vs_featured")
print_global_loss(df_feat_cmp, "Perdida map-matched -> featured")
display(df_feat_cmp.sort_values("loss_pct", ascending=False).head(20))
"""


NOTEBOOK_PHASE4_PLOTS_CODE = """featured_summaries = []
all_feature_vars = {}
all_feature_vars.update(BASE_VARIABLES)
all_feature_vars.update(FEATURE_VARIABLES)

for var_name, aliases in all_feature_vars.items():
    print("=" * 90)
    print(f"Variable: {var_name} (FEATURED)")
    featured_summaries.append(plot_variable_distribution(featured_df, var_name, aliases, "featured"))

featured_summary_df = pd.DataFrame(featured_summaries)
print("\\nResumen FEATURED")
display(featured_summary_df)
"""


NOTEBOOK_CLOSING_MD = """## Resumen final

Recomendaciones para modelado:

- Revisar variables con alto `% invalid_pct` antes de entrenar.
- Si hay `% outlier_pct_over_valid` muy alto, evaluar clipping/winsorization o escalado robusto.
- Si la distancia de map-matching muestra cola larga, revisar segmentos problematicos antes del entrenamiento final.
- Guardar tablas de resumen por etapa como CSV para trazabilidad entre experimentos.
"""


def mk_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def mk_code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def build_notebook() -> dict:
    cells = [
        mk_markdown_cell(NOTEBOOK_INTRO_MD),
        mk_code_cell(NOTEBOOK_SETUP_CODE),
        mk_code_cell(NOTEBOOK_HELPERS_CODE),
        mk_markdown_cell(NOTEBOOK_PHASE2_MD),
        mk_code_cell(NOTEBOOK_PHASE2_CODE),
        mk_code_cell(NOTEBOOK_PHASE2_PLOTS_CODE),
        mk_markdown_cell(NOTEBOOK_PHASE3_MD),
        mk_code_cell(NOTEBOOK_PHASE3_CODE),
        mk_code_cell(NOTEBOOK_PHASE3_PLOTS_CODE),
        mk_markdown_cell(NOTEBOOK_PHASE4_MD),
        mk_code_cell(NOTEBOOK_PHASE4_CODE),
        mk_code_cell(NOTEBOOK_PHASE4_PLOTS_CODE),
        mk_markdown_cell(NOTEBOOK_CLOSING_MD),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EDA notebook for LiDAR stability data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-notebook",
        default="notebooks/eda_pipeline_report.ipynb",
        help="Output notebook path relative to repo root.",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()
    out_path = (repo_root / args.output_notebook).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Notebook generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

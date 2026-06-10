#!/usr/bin/env python3
"""Analyze IMU rows dropped as outliers for the featured dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


IMU_AXES = ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"]
OUTLIER_REASON = "imu_rolling_outlier"


def source_key_from_featured_csv(path: Path) -> str:
    """Map segmented featured files back to their daily outlier audit source_key."""
    return re.sub(r"_seg\d+$", "", path.stem)


def load_featured_source_keys(featured_dir: Path) -> set[str]:
    return {
        source_key_from_featured_csv(path)
        for path in featured_dir.rglob("*.csv")
        if path.is_file()
    }


def read_imu_denominator(summary_path: Path) -> int:
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    return sum(
        int(step.get("before_rows", 0))
        for step in summary.get("filter_steps", [])
        if step.get("stage") == "stability"
        and step.get("filter_name") == OUTLIER_REASON
    )


def count_flagged_axes(dropped_rows_path: Path) -> tuple[dict[str, int], int]:
    counts = {axis: 0 for axis in IMU_AXES}
    dropped_rows = 0

    if dropped_rows_path.stat().st_size == 0:
        return counts, dropped_rows

    usecols = lambda col: col in {"stage", "reason", "flagged_axes"}
    df = pd.read_csv(dropped_rows_path, usecols=usecols)
    if df.empty or "flagged_axes" not in df.columns:
        return counts, dropped_rows

    imu_df = df[
        (df["stage"] == "stability")
        & (df["reason"] == OUTLIER_REASON)
        & df["flagged_axes"].notna()
    ]
    dropped_rows = len(imu_df)

    for axes_text in imu_df["flagged_axes"].astype(str):
        for axis in axes_text.split("|"):
            axis = axis.strip()
            if axis in counts:
                counts[axis] += 1

    return counts, dropped_rows


def build_summary(featured_dir: Path, outliers_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    source_keys = load_featured_source_keys(featured_dir)
    if not source_keys:
        raise FileNotFoundError(f"No CSV files found under {featured_dir}")

    axis_counts = {axis: 0 for axis in IMU_AXES}
    total_before_rows = 0
    total_dropped_rows = 0
    matched_sources = 0

    for source_key in sorted(source_keys):
        summary_path = outliers_dir / f"{source_key}_drop_summary.json"
        dropped_rows_path = outliers_dir / f"{source_key}_dropped_rows.csv"

        if not summary_path.exists() or not dropped_rows_path.exists():
            continue

        matched_sources += 1
        total_before_rows += read_imu_denominator(summary_path)
        file_counts, file_dropped_rows = count_flagged_axes(dropped_rows_path)
        total_dropped_rows += file_dropped_rows
        for axis, count in file_counts.items():
            axis_counts[axis] += count

    if matched_sources == 0:
        raise FileNotFoundError(
            f"No matching outlier audit files found in {outliers_dir} "
            f"for featured source keys from {featured_dir}"
        )

    total_axis_flags = sum(axis_counts.values())
    rows = []
    for axis in IMU_AXES:
        flagged_rows = axis_counts[axis]
        rows.append(
            {
                "imu_variable": axis,
                "flagged_rows": flagged_rows,
                "pct_over_total_imu_rows": (
                    flagged_rows / total_before_rows * 100 if total_before_rows else 0.0
                ),
                "pct_of_axis_outlier_flags": (
                    flagged_rows / total_axis_flags * 100 if total_axis_flags else 0.0
                ),
            }
        )

    metadata = {
        "featured_csv_files": len(list(featured_dir.rglob("*.csv"))),
        "featured_source_keys": len(source_keys),
        "matched_outlier_sources": matched_sources,
        "total_imu_rows_before_filter": total_before_rows,
        "total_rows_dropped_by_imu_filter": total_dropped_rows,
        "total_axis_flags": total_axis_flags,
    }
    return pd.DataFrame(rows), metadata


def plot_pie(summary_df: pd.DataFrame, metadata: dict[str, int], output_png: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_png.parent / "matplotlib_cache"))

    import matplotlib.pyplot as plt

    plot_df = summary_df[summary_df["flagged_rows"] > 0].copy()
    if plot_df.empty:
        raise ValueError("No IMU outlier flags found to plot")

    plot_df = plot_df.sort_values("flagged_rows", ascending=False)
    labels = [
        f"{row.imu_variable}\n{row.pct_over_total_imu_rows:.2f}% datos"
        for row in plot_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = plt.get_cmap("Set2").colors
    wedges, texts, autotexts = ax.pie(
        plot_df["flagged_rows"],
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        labeldistance=1.08,
        colors=colors[: len(plot_df)],
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for text in autotexts:
        text.set_fontsize(10)
        text.set_weight("bold")

    dropped_pct = (
        metadata["total_rows_dropped_by_imu_filter"]
        / metadata["total_imu_rows_before_filter"]
        * 100
        if metadata["total_imu_rows_before_filter"]
        else 0.0
    )
    ax.set_title(
        "Datos perdidos por outliers IMU en featured\n"
        f"{metadata['total_rows_dropped_by_imu_filter']:,} filas eliminadas "
        f"({dropped_pct:.2f}% de {metadata['total_imu_rows_before_filter']:,})",
        fontsize=14,
        pad=18,
    )
    ax.axis("equal")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pie chart of IMU outlier data loss for featured files."
    )
    parser.add_argument(
        "--featured-dir",
        type=Path,
        default=Path("Doback-Data/featured"),
        help="Directory with featured CSV files.",
    )
    parser.add_argument(
        "--outliers-dir",
        type=Path,
        default=Path("Doback-Data/processed-data/outliers"),
        help="Directory with *_dropped_rows.csv and *_drop_summary.json audit files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/imu_outlier_analysis"),
        help="Directory where the PNG and CSV summary will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, metadata = build_summary(args.featured_dir, args.outliers_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "featured_imu_outlier_loss_summary.csv"
    png_path = args.output_dir / "featured_imu_outlier_loss_pie.png"
    metadata_path = args.output_dir / "featured_imu_outlier_loss_metadata.json"

    summary_df.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_pie(summary_df, metadata, png_path)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {metadata_path}")
    print(summary_df.sort_values("flagged_rows", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a graphical leaderboard from saved model metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ranking and plots for all trained models in a metrics folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-dir",
        default="src/lidar_stability/ml/models",
        help="Folder containing *_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output folder for leaderboard CSV/JSON and plots. Defaults to metrics-dir.",
    )
    parser.add_argument(
        "--title",
        default="Model Leaderboard (w prediction)",
        help="Title used in generated figures",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=0,
        help="Optional minimum n_samples filter",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not {"model", "rmse_mean", "mae_mean", "r2_mean"}.issubset(payload.keys()):
            continue

        rows.append(
            {
                "model": payload.get("model", path.stem),
                "rmse_mean": float(payload.get("rmse_mean")),
                "mae_mean": float(payload.get("mae_mean")),
                "r2_mean": float(payload.get("r2_mean")),
                "n_samples": int(payload.get("n_samples", 0)),
                "n_features": int(payload.get("n_features", 0)),
                "metrics_path": str(path),
                "model_path": payload.get("model_path", ""),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No valid *_metrics.json files found in: {metrics_dir}")

    return pd.DataFrame(rows)


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()

    ranked["rank_rmse"] = ranked["rmse_mean"].rank(method="min", ascending=True)
    ranked["rank_mae"] = ranked["mae_mean"].rank(method="min", ascending=True)
    ranked["rank_r2"] = ranked["r2_mean"].rank(method="min", ascending=False)

    ranked["rank_score"] = (ranked["rank_rmse"] + ranked["rank_mae"] + ranked["rank_r2"]) / 3.0
    ranked["overall_rank"] = ranked["rank_score"].rank(method="min", ascending=True).astype(int)

    ranked = ranked.sort_values(["overall_rank", "rank_score", "rmse_mean", "mae_mean"], ascending=[True, True, True, True])
    return ranked


def save_tables(ranked: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "leaderboard_table.csv"
    json_path = out_dir / "leaderboard_table.json"

    ranked.to_csv(csv_path, index=False)
    json_path.write_text(ranked.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path


def plot_leaderboard(ranked: pd.DataFrame, out_dir: Path, title: str) -> list[Path]:
    sns.set_theme(style="whitegrid")
    outputs: list[Path] = []

    # Plot 1: Overall rank score (lower is better)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ranked, x="model", y="rank_score", hue="model", palette="crest", legend=False)
    plt.title(f"{title} - Overall rank score (lower is better)")
    plt.xlabel("Model")
    plt.ylabel("Rank score")
    plt.tight_layout()
    p1 = out_dir / "leaderboard_rank_score.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    outputs.append(p1)

    # Plot 2: Metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.barplot(data=ranked, x="model", y="rmse_mean", hue="model", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("RMSE (lower is better)")

    sns.barplot(data=ranked, x="model", y="mae_mean", hue="model", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title("MAE (lower is better)")

    sns.barplot(data=ranked, x="model", y="r2_mean", hue="model", ax=axes[2], palette="Reds_d", legend=False)
    axes[2].set_title("R2 (higher is better)")

    for ax in axes:
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(title)
    plt.tight_layout()
    p2 = out_dir / "leaderboard_metrics.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    outputs.append(p2)

    return outputs


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()

    metrics_dir = (repo_root / args.metrics_dir).resolve()
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}")

    out_dir = (repo_root / args.output_dir).resolve() if args.output_dir else metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_dir)

    if args.min_samples > 0:
        df = df[df["n_samples"] >= int(args.min_samples)].copy()
        if df.empty:
            raise RuntimeError("No models left after applying --min-samples filter")

    ranked = rank_models(df)
    csv_path, json_path = save_tables(ranked, out_dir)
    plot_paths = plot_leaderboard(ranked, out_dir, args.title)

    print("Leaderboard generated")
    print(f"  models: {len(ranked)}")
    print("  ranking:")
    for _, row in ranked.iterrows():
        print(
            f"    #{row['overall_rank']} {row['model']} | "
            f"RMSE={row['rmse_mean']:.6f} MAE={row['mae_mean']:.6f} R2={row['r2_mean']:.4f}"
        )

    print(f"  table CSV: {csv_path}")
    print(f"  table JSON: {json_path}")
    for p in plot_paths:
        print(f"  plot: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


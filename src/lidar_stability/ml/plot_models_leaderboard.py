#!/usr/bin/env python3
"""Build a graphical leaderboard from saved model metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ranking and plots for all trained models in a metrics folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-dir",
        default="output/models",
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
    parser.add_argument(
        "--html-file",
        default="leaderboard.html",
        help="Output HTML filename for interactive leaderboard report",
    )
    parser.add_argument(
        "--compact-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only writes HTML + JSON table (fewer files).",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    def _to_row(entry: dict, path: Path) -> dict | None:
        if not {"model", "rmse_mean", "mae_mean", "r2_mean"}.issubset(entry.keys()):
            return None
        return {
            "model": entry.get("model", path.stem),
            "run_id": entry.get("run_id", "default"),
            "rmse_mean": float(entry.get("rmse_mean")),
            "mae_mean": float(entry.get("mae_mean")),
            "r2_mean": float(entry.get("r2_mean")),
            "n_samples": int(entry.get("n_samples", 0)),
            "n_features": int(entry.get("n_features", 0)),
            "metrics_path": str(path),
            "model_path": entry.get("model_path", ""),
        }

    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            for entry in payload["runs"]:
                if not isinstance(entry, dict):
                    continue
                row = _to_row(entry, path)
                if row is not None:
                    rows.append(row)
            continue

        if isinstance(payload, dict):
            row = _to_row(payload, path)
            if row is not None:
                rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No valid *_metrics.json files found in: {metrics_dir}")

    return pd.DataFrame(rows)


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["label"] = ranked.apply(
        lambda row: row["model"] if str(row["run_id"]) == "default" else f"{row['model']}@{row['run_id']}",
        axis=1,
    )

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
    sns.barplot(data=ranked, x="label", y="rank_score", hue="model", palette="crest", legend=False)
    plt.title(f"{title} - Overall rank score (lower is better)")
    plt.xlabel("Model run")
    plt.ylabel("Rank score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p1 = out_dir / "leaderboard_rank_score.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    outputs.append(p1)

    # Plot 2: Metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.barplot(data=ranked, x="label", y="rmse_mean", hue="model", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("RMSE (lower is better)")

    sns.barplot(data=ranked, x="label", y="mae_mean", hue="model", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title("MAE (lower is better)")

    sns.barplot(data=ranked, x="label", y="r2_mean", hue="model", ax=axes[2], palette="Reds_d", legend=False)
    axes[2].set_title("R2 (higher is better)")

    for ax in axes:
        ax.set_xlabel("Model run")
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle(title)
    plt.tight_layout()
    p2 = out_dir / "leaderboard_metrics.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    outputs.append(p2)

    return outputs


def save_html_report(ranked: pd.DataFrame, out_dir: Path, title: str, html_file: str) -> Path:
    html_path = out_dir / html_file

    rank_fig = px.bar(
        ranked,
        x="label",
        y="rank_score",
        color="model",
        title=f"{title} - Overall rank score (lower is better)",
        labels={"label": "Model run", "rank_score": "Rank score"},
    )
    rank_fig.update_layout(xaxis_tickangle=-25)

    metrics_fig = go.Figure()
    metrics_fig.add_trace(
        go.Bar(name="RMSE", x=ranked["label"], y=ranked["rmse_mean"], marker_color="#1f77b4")
    )
    metrics_fig.add_trace(
        go.Bar(name="MAE", x=ranked["label"], y=ranked["mae_mean"], marker_color="#2ca02c")
    )
    metrics_fig.add_trace(
        go.Bar(name="R2", x=ranked["label"], y=ranked["r2_mean"], marker_color="#d62728")
    )
    metrics_fig.update_layout(
        barmode="group",
        title=f"{title} - Metrics comparison",
        xaxis_title="Model run",
        yaxis_title="Metric value",
        xaxis_tickangle=-25,
    )

    table_df = ranked[
        [
            "overall_rank",
            "label",
            "model",
            "run_id",
            "rmse_mean",
            "mae_mean",
            "r2_mean",
            "n_samples",
            "n_features",
        ]
    ].copy()
    table_html = table_df.to_html(index=False, float_format=lambda x: f"{x:.6f}")

    rank_html = rank_fig.to_html(full_html=False, include_plotlyjs="cdn")
    metrics_html = metrics_fig.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ margin-top: 0; color: #555; }}
    .card {{ margin: 20px 0; padding: 16px; border: 1px solid #ddd; border-radius: 8px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Auto-generated comparison from *_metrics.json files in the selected folder.</p>
  <div class=\"card\">{rank_html}</div>
  <div class=\"card\">{metrics_html}</div>
  <div class=\"card\"><h2>Ranking table</h2>{table_html}</div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


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
    csv_path = None
    plot_paths: list[Path] = []
    if not args.compact_output:
        csv_path, _ = save_tables(ranked, out_dir)
        plot_paths = plot_leaderboard(ranked, out_dir, args.title)

    json_path = out_dir / "leaderboard_table.json"
    json_path.write_text(ranked.to_json(orient="records"), encoding="utf-8")
    html_path = save_html_report(ranked, out_dir, args.title, args.html_file)

    print("Leaderboard generated")
    print(f"  models: {len(ranked)}")
    print("  ranking:")
    for _, row in ranked.iterrows():
        print(
            f"    #{row['overall_rank']} {row['label']} | "
            f"RMSE={row['rmse_mean']:.6f} MAE={row['mae_mean']:.6f} R2={row['r2_mean']:.4f}"
        )

    if csv_path is not None:
        print(f"  table CSV: {csv_path}")
    print(f"  table JSON: {json_path}")
    print(f"  report HTML: {html_path}")
    for p in plot_paths:
        print(f"  plot: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


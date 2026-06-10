"""Load model metrics into a sortable leaderboard dataframe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _run_to_row(run: dict, source: Path) -> dict:
    return {
        "model": run.get("model") or run.get("model_name") or source.stem,
        "rmse_mean": run.get("rmse_mean") or run.get("holdout_rmse") or run.get("cv_rmse_mean"),
        "mae_mean": run.get("mae_mean") or run.get("holdout_mae") or run.get("cv_mae_mean"),
        "r2_mean": run.get("r2_mean") or run.get("holdout_r2") or run.get("cv_r2_mean"),
        "source_file": str(source),
    }


def load_metrics(root: str | Path, enrich_from_bundle: bool = True) -> pd.DataFrame:
    """Load metrics JSON files written by training/search commands."""
    root_path = Path(root)
    rows: list[dict] = []
    metric_files = sorted(root_path.glob("*metrics.json")) + sorted(root_path.glob("*history.json"))
    metric_files += sorted(root_path.glob("*leaderboard.json"))
    for metrics_path in metric_files:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(_run_to_row(item, metrics_path) for item in payload)
        elif isinstance(payload.get("runs"), list):
            rows.extend(_run_to_row(run, metrics_path) for run in payload["runs"])
        elif "history" in payload and payload["history"]:
            rows.extend(_run_to_row(item, metrics_path) for item in payload["history"])
        else:
            rows.append(_run_to_row(payload, metrics_path))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True], na_position="last").reset_index(drop=True)

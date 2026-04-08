#!/usr/bin/env python3
"""Build a graphical leaderboard from saved model metrics and leaderboard JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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
        default="output/models",
        help="Folder containing *_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="output/leaderboard",
        help="Optional output folder for leaderboard CSV/JSON and plots. Defaults to output/leaderboard.",
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
    parser.add_argument(
        "--enrich-from-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, reads model bundles (.joblib) to backfill missing params/n_features. Disable for faster execution.",
    )
    parser.add_argument(
        "--dedupe-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, keep only one entry per model label (model@run_id), choosing the best metrics.",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_metrics(metrics_dir: Path, enrich_from_bundle: bool = True) -> pd.DataFrame:
    bundle_cache: dict[Path, dict | None] = {}

    def _get_bundle(path: Path) -> dict | None:
        if path in bundle_cache:
            return bundle_cache[path]
        try:
            loaded = joblib.load(path)
            bundle_cache[path] = loaded if isinstance(loaded, dict) else None
        except Exception:
            bundle_cache[path] = None
        return bundle_cache[path]

    def _parse_bundle_ref(model_path: str) -> tuple[Path | None, str | None]:
        if not model_path:
            return None, None
        if "::" in model_path:
            path_str, model_key = model_path.split("::", 1)
            return Path(path_str), model_key
        return Path(model_path), None

    def _enrich_from_bundle(row: dict) -> dict:
        bundle_path, model_key = _parse_bundle_ref(str(row.get("model_path") or ""))
        if bundle_path is None:
            return row

        bundle = _get_bundle(bundle_path)
        if not isinstance(bundle, dict):
            return row

        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return row

        artifact = None
        if model_key and model_key in artifacts:
            artifact = artifacts.get(model_key)
        else:
            label = f"{row.get('model')}@{row.get('run_id')}"
            artifact = artifacts.get(label)
            if artifact is None and len(artifacts) == 1:
                artifact = next(iter(artifacts.values()))

        if not isinstance(artifact, dict):
            return row

        if int(row.get("n_features", 0) or 0) == 0:
            feature_columns = artifact.get("feature_columns")
            if isinstance(feature_columns, list):
                row["n_features"] = len(feature_columns)

        params_raw = row.get("params", "{}")
        if params_raw in (None, "", "{}"):
            model_obj = artifact.get("model")
            if model_obj is not None and hasattr(model_obj, "get_params"):
                try:
                    row["params"] = json.dumps(model_obj.get_params(deep=False), ensure_ascii=True)
                except Exception:
                    pass

        return row

    def _to_row(entry: dict, path: Path) -> dict | None:
        if not {"model", "rmse_mean", "mae_mean", "r2_mean"}.issubset(entry.keys()):
            return None

        n_samples = int(entry.get("n_samples", 0) or 0)
        n_features = int(entry.get("n_features", 0) or 0)
        if n_features == 0 and isinstance(entry.get("feature_columns"), list):
            n_features = len(entry.get("feature_columns", []))

        raw_params = entry.get("params")
        if isinstance(raw_params, dict):
            params_text = json.dumps(raw_params, ensure_ascii=True)
        elif isinstance(raw_params, str) and raw_params.strip():
            params_text = raw_params
        else:
            run_cfg = entry.get("run_config")
            if isinstance(run_cfg, dict) and isinstance(run_cfg.get("hyperparameters"), dict):
                params_text = json.dumps(run_cfg["hyperparameters"], ensure_ascii=True)
            else:
                params_text = "{}"

        row = {
            "model": entry.get("model", path.stem),
            "run_id": entry.get("run_id", "default"),
            "rmse_mean": float(entry.get("rmse_mean")),
            "mae_mean": float(entry.get("mae_mean")),
            "r2_mean": float(entry.get("r2_mean")),
            "n_samples": n_samples,
            "n_features": n_features,
            "metrics_path": str(path),
            "model_path": entry.get("model_path", ""),
            "params": params_text,
        }

        needs_enrichment = enrich_from_bundle and (
            int(row.get("n_features", 0) or 0) == 0 or row.get("params") in (None, "", "{}")
        )
        return _enrich_from_bundle(row) if needs_enrichment else row

    def _history_to_row(entry: dict, payload: dict, path: Path) -> dict | None:
        if not {"cv_rmse_mean", "cv_mae_mean", "cv_r2_mean"}.issubset(entry.keys()):
            return None

        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}

        feature_columns = dataset.get("feature_columns", config.get("feature_columns", []))
        n_features = int(dataset.get("n_features", 0) or 0)
        if n_features == 0 and isinstance(feature_columns, list):
            n_features = len(feature_columns)

        n_samples = int(dataset.get("n_samples", 0) or config.get("n_samples", 0) or 0)
        if n_samples == 0:
            max_rows = int(config.get("max_rows", 0) or 0)
            if max_rows > 0:
                n_samples = max_rows
            else:
                input_files = config.get("input_glob")
                max_rows_per_source = int(config.get("max_rows_per_source", 0) or 0)
                if isinstance(input_files, list) and input_files:
                    if max_rows_per_source > 0:
                        n_samples = len(input_files) * max_rows_per_source
                    else:
                        n_samples = len(input_files)

        row = {
            "model": entry.get("model") or config.get("model", path.stem),
            "run_id": entry.get("trial", "default"),
            "rmse_mean": float(entry.get("cv_rmse_mean")),
            "mae_mean": float(entry.get("cv_mae_mean")),
            "r2_mean": float(entry.get("cv_r2_mean")),
            "n_samples": n_samples,
            "n_features": n_features,
            "metrics_path": str(path),
            "model_path": payload.get("model_path", ""),
            "params": json.dumps(entry.get("params", {}), ensure_ascii=True) if isinstance(entry.get("params", {}), dict) else "{}",
        }

        needs_enrichment = enrich_from_bundle and (
            int(row.get("n_features", 0) or 0) == 0 or row.get("params") in (None, "", "{}")
        )
        return _enrich_from_bundle(row) if needs_enrichment else row

    rows = []
    candidate_paths = (
        list(metrics_dir.glob("*_metrics.json"))
        + list(metrics_dir.glob("*_leaderboard.json"))
        + list(metrics_dir.glob("*_history.json"))
    )

    for path in sorted({path.resolve(): path for path in candidate_paths}.values(), key=lambda item: item.name):
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

        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            for entry in payload["history"]:
                if not isinstance(entry, dict):
                    continue
                row = _history_to_row(entry, payload, path)
                if row is not None:
                    rows.append(row)
            continue

        if isinstance(payload, list):
            for entry in payload:
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
        raise FileNotFoundError(
            f"No valid *_metrics.json, *_leaderboard.json, or *_history.json files found in: {metrics_dir}"
        )

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


def dedupe_by_label(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    deduped = df.copy()
    deduped["label"] = deduped.apply(
        lambda row: row["model"] if str(row["run_id"]) == "default" else f"{row['model']}@{row['run_id']}",
        axis=1,
    )
    before = len(deduped)

    # Keep the strongest variant per logical model label.
    deduped = deduped.sort_values(
        ["label", "r2_mean", "rmse_mean", "mae_mean", "n_samples", "n_features"],
        ascending=[True, False, True, True, False, False],
    )
    deduped = deduped.drop_duplicates(subset=["label"], keep="first").copy()
    removed = before - len(deduped)
    return deduped.drop(columns=["label"]), removed


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

    def _pretty_params_text(value: str) -> str:
        text = str(value or "").strip()
        if not text or text == "{}":
            return "(sin parametros)"

        # Some rows may contain JSON serialized as string literals.
        for _ in range(2):
            if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                try:
                    text = json.loads(text)
                except Exception:
                    break
            if isinstance(text, str):
                text = text.strip()

        if isinstance(text, dict):
            payload = text
        else:
            try:
                payload = json.loads(text)
            except Exception:
                return str(value)

        if not isinstance(payload, dict) or not payload:
            return "(sin parametros)"

        return "\n".join(f"{k}: {payload[k]}" for k in sorted(payload.keys()))

    def _params_cell(value: str) -> str:
        pretty = _pretty_params_text(value)
        safe = (
            pretty.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return f"<pre class=\"params\">{safe}</pre>"

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
            "params",
        ]
    ].copy()
    table_df["params"] = table_df["params"].map(_params_cell)
    table_html = table_df.to_html(index=False, float_format=lambda x: f"{x:.6f}", escape=False)

    glossary_html = """
<div class=\"card\">
    <h2>Glosario de hiperparametros</h2>
    <p>Esta guia resume el efecto de cada parametro y cuando conviene subirlo o bajarlo.</p>
    <table>
        <thead>
            <tr>
                <th>Parametro</th>
                <th>Que controla</th>
                <th>Subirlo suele implicar</th>
                <th>Bajarlo suele implicar</th>
                <th>Cuando tocarlo</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>n_estimators</td>
                <td>Numero de arboles del ensamble</td>
                <td>Mas estabilidad y menos varianza, mas tiempo/memoria</td>
                <td>Entrenamiento mas rapido, algo mas de ruido</td>
                <td>Sube si la metrica oscila entre folds; baja si el coste es alto y ya estas en meseta</td>
            </tr>
            <tr>
                <td>max_depth</td>
                <td>Profundidad maxima de cada arbol</td>
                <td>Modelos mas expresivos, mayor riesgo de sobreajuste</td>
                <td>Mas regularizacion, posible infraajuste</td>
                <td>Baja si train muy alto y validacion cae; sube si no captura no linealidades</td>
            </tr>
            <tr>
                <td>min_samples_leaf</td>
                <td>Muestras minimas en hoja</td>
                <td>Hojas mas grandes, menos ruido, fronteras mas suaves</td>
                <td>Hojas pequenas, mas detalle, mas varianza</td>
                <td>Sube con datos ruidosos; baja si pierdes detalle local</td>
            </tr>
            <tr>
                <td>min_samples_split</td>
                <td>Muestras minimas para dividir un nodo</td>
                <td>Menos divisiones, arboles mas simples</td>
                <td>Mas divisiones, arboles mas complejos</td>
                <td>Sube para contener sobreajuste; baja si hay infraajuste</td>
            </tr>
            <tr>
                <td>max_features</td>
                <td>Numero/fraccion de variables evaluadas por split</td>
                <td>Menos diversidad entre arboles, a veces mejor ajuste puntual</td>
                <td>Mas diversidad del ensamble, mayor robustez</td>
                <td>Baja si los arboles se parecen demasiado; sube si falta capacidad</td>
            </tr>
            <tr>
                <td>learning_rate (GBR)</td>
                <td>Peso de cada etapa de boosting</td>
                <td>Convergencia rapida, mas riesgo de sobreajuste</td>
                <td>Aprendizaje mas gradual y estable</td>
                <td>Si lo bajas, normalmente compensa subiendo n_estimators</td>
            </tr>
            <tr>
                <td>subsample (GBR)</td>
                <td>Fraccion de muestras por iteracion</td>
                <td>Con 1.0: menos ruido estocastico, mas riesgo de sobreajuste</td>
                <td>Mas estocastico, mejor generalizacion en muchos casos</td>
                <td>Prueba 0.6-0.9 para mejorar generalizacion cuando hay mucho dato</td>
            </tr>
            <tr>
                <td>bootstrap / oob_score (RF)</td>
                <td>Muestreo con reemplazo y validacion OOB</td>
                <td>Con bootstrap=true hay mas diversidad de arboles</td>
                <td>Sin bootstrap, arboles mas correlacionados</td>
                <td>Activa oob_score para tener una referencia adicional sin CV extra</td>
            </tr>
            <tr>
                <td>ccp_alpha</td>
                <td>Intensidad de poda por coste-complejidad</td>
                <td>Mas poda, menor complejidad</td>
                <td>Menos poda, mayor capacidad</td>
                <td>Sube ligeramente si detectas sobreajuste persistente</td>
            </tr>
        </tbody>
    </table>
    <p><strong>Regla practica:</strong> ajusta de uno en uno y valida con los mismos folds para comparar de forma justa.</p>
</div>
"""

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
        .params {{ margin: 0; white-space: pre-wrap; text-align: left; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
    <p>Auto-generated comparison from *_metrics.json, *_leaderboard.json and *_history.json files in the selected folder.</p>
  <div class=\"card\"><h2>Ranking table</h2>{table_html}</div>
    {glossary_html}
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

    df = load_metrics(metrics_dir, enrich_from_bundle=args.enrich_from_bundle)

    if args.min_samples > 0:
        df = df[df["n_samples"] >= int(args.min_samples)].copy()
        if df.empty:
            raise RuntimeError("No models left after applying --min-samples filter")

    removed_duplicates = 0
    if args.dedupe_labels:
        df, removed_duplicates = dedupe_by_label(df)
        if df.empty:
            raise RuntimeError("No models left after deduplication")

    ranked = rank_models(df)
    csv_path = None
    if not args.compact_output:
        csv_path, _ = save_tables(ranked, out_dir)

    json_path = out_dir / "leaderboard_table.json"
    json_path.write_text(ranked.to_json(orient="records"), encoding="utf-8")
    html_path = save_html_report(ranked, out_dir, args.title, args.html_file)

    print("Leaderboard generated")
    print(f"  models: {len(ranked)}")
    if args.dedupe_labels:
        print(f"  duplicates removed: {removed_duplicates}")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


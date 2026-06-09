#!/usr/bin/env python3
"""Genera gráficas comparativas entre modelos en una carpeta de modelos.

Busca recursivamente subcarpetas bajo `root_dir` (por defecto `output/models`) y extrae
`holdout_r2`, `holdout_rmse` y `generalization_gap` desde archivos `leaderboard` (CSV)
o `_history.json`. Guarda tres gráficas de barras en `out_dir`.

Uso:
    python tmp/compare_models_metrics.py --root output/models --out tmp/figs

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_from_leaderboard(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    keys = ['holdout_r2', 'holdout_rmse', 'generalization_gap', 'objective_value', 'cv_r2_mean']
    res = {k: None for k in keys}
    # Normalizar nombres de columnas a lower
    cols = {c.lower(): c for c in df.columns}
    # Preferir fila mejor por holdout_r2, sino por objective_value (may need ajuste)
    sel_row = None
    if 'holdout_r2' in cols:
        idx = df[cols['holdout_r2']].idxmax()
        sel_row = df.loc[idx]
    elif 'objective_value' in cols:
        idx = df[cols['objective_value']].idxmax()
        sel_row = df.loc[idx]
    else:
        sel_row = df.iloc[0]

    for k in ['holdout_r2', 'holdout_rmse', 'generalization_gap', 'objective_value', 'cv_r2_mean']:
        if k in cols:
            try:
                res[k] = float(sel_row[cols[k]])
            except Exception:
                res[k] = None
    return res


def extract_from_history(path: Path) -> Dict[str, Optional[float]]:
    res = {'holdout_r2': None, 'holdout_rmse': None, 'generalization_gap': None, 'cv_r2_mean': None}
    try:
        with open(path, 'r') as fh:
            h = json.load(fh)
    except Exception:
        return res

    trials = []
    if isinstance(h, dict) and 'history' in h and isinstance(h['history'], list):
        trials = h['history']
    elif isinstance(h, list):
        trials = h

    if not trials:
        return res

    # Buscar mejor trial por holdout_r2 o por cv_r2
    best = None
    for t in trials:
        if not isinstance(t, dict):
            continue
        if best is None:
            best = t
            continue
        # preferir mayor holdout_r2
        if 'holdout_r2' in t and 'holdout_r2' in best:
            try:
                if float(t['holdout_r2']) > float(best['holdout_r2']):
                    best = t
            except Exception:
                pass

    if best is None:
        return res

    for k in res.keys():
        if k in best:
            try:
                res[k] = float(best[k])
            except Exception:
                res[k] = None

    # si falta generalization_gap, intentar calcular desde cv_r2_mean - holdout_r2
    if res.get('generalization_gap') is None and res.get('cv_r2_mean') is not None and res.get('holdout_r2') is not None:
        res['generalization_gap'] = float(res['cv_r2_mean']) - float(res['holdout_r2'])

    return res


def scan_models(root_dir: Path) -> pd.DataFrame:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        p = Path(dirpath)
        # Skip the root itself if empty
        # Detect candidate model folders: those that contain any *_leaderboard*.csv or *_history.json or *_report.md
        found = False
        for name in filenames:
            ln = name.lower()
            if 'leaderboard' in ln or ln.endswith('_history.json') or ln.endswith('_report.md') or 'holdout' in ln:
                found = True
                break
        if not found:
            continue

        metrics = {'holdout_r2': None, 'holdout_rmse': None, 'generalization_gap': None}

        # try leaderboard csv
        lb_files = list(p.glob('*leaderboard*.csv'))
        if lb_files:
            try:
                df = pd.read_csv(lb_files[0], low_memory=False)
                ext = extract_from_leaderboard(df)
                for k in metrics:
                    metrics[k] = ext.get(k) or metrics[k]
            except Exception:
                pass

        # try history json
        hist_files = list(p.glob('*_history.json'))
        if hist_files:
            ext = extract_from_history(hist_files[0])
            for k in metrics:
                if metrics[k] is None:
                    metrics[k] = ext.get(k)

        # as last resource, check for leaderboard-like single-row csv files that may contain direct columns
        if all(v is None for v in metrics.values()):
            # try all csv files in folder
            for f in p.glob('*.csv'):
                try:
                    df = pd.read_csv(f, nrows=5, low_memory=False)
                    cols = {c.lower() for c in df.columns}
                    if {'holdout_r2', 'holdout_rmse'}.intersection(cols) or 'generalization_gap' in cols:
                        ext = extract_from_leaderboard(df)
                        for k in metrics:
                            if metrics[k] is None:
                                metrics[k] = ext.get(k)
                        break
                except Exception:
                    continue

        rows.append({
            'model_dir': os.path.relpath(str(p), str(root_dir)),
            'full_path': str(p),
            'holdout_r2': metrics['holdout_r2'],
            'holdout_rmse': metrics['holdout_rmse'],
            'generalization_gap': metrics['generalization_gap'],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # normalize types
    for c in ['holdout_r2', 'holdout_rmse', 'generalization_gap']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, ascending: bool = False):
    dfp = df.dropna(subset=[metric]).copy()
    if dfp.empty:
        print(f'No hay valores para {metric}, se salta la gráfica.')
        return
    dfp = dfp.sort_values(metric, ascending=ascending)
    plt.figure(figsize=(10, max(4, 0.25 * len(dfp))))
    sns.set(style='whitegrid')
    ax = sns.barplot(x=metric, y='model_dir', data=dfp, palette='viridis')
    ax.set_xlabel(metric)
    ax.set_ylabel('Model (relative path)')
    ax.set_title(f'Comparación: {metric}')

    # Mostrar el valor al final de cada barra para facilitar la lectura rápida.
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f'Guardada gráfica: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='output/models', help='Carpeta raíz donde buscar modelos')
    parser.add_argument('--out', default='src/lidar_stability/tmp/figs', help='Carpeta donde guardar las gráficas')
    parser.add_argument('--show', action='store_true', help='Mostrar las figuras además de guardarlas')
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        print(f'Error: root no existe: {root}')
        return

    print(f'Escaneando modelos en: {root} ...')
    df = scan_models(root)
    if df.empty:
        print('No se encontraron carpetas de modelos con artefactos reconocibles.')
        return

    print('Métricas recogidas:')
    print(df[['model_dir', 'holdout_r2', 'holdout_rmse', 'generalization_gap']].to_string(index=False))

    out.mkdir(parents=True, exist_ok=True)
    plot_metric(df, 'holdout_r2', out / 'holdout_r2.png', ascending=False)
    plot_metric(df, 'holdout_rmse', out / 'holdout_rmse.png', ascending=True)
    plot_metric(df, 'generalization_gap', out / 'generalization_gap.png', ascending=False)

    if args.show:
        # mostrar imágenes usando matplotlib (reabrir y mostrar)
        for fname in ['holdout_r2.png', 'holdout_rmse.png', 'generalization_gap.png']:
            p = out / fname
            if p.exists():
                img = plt.imread(str(p))
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.show()


if __name__ == '__main__':
    main()

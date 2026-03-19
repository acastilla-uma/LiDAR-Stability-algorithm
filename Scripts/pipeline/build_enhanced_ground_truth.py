#!/usr/bin/env python3
"""CLI to generate sprint-5 enhanced ground truth CSVs from featured data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from physics import StabilityEngine
from pipeline.ground_truth import build_enhanced_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build enhanced ground truth from featured CSV files')
    parser.add_argument('--input-glob', default='Doback-Data/featured/DOBACK*.csv')
    parser.add_argument('--config', default='Scripts/config/vehicle.yaml')
    parser.add_argument('--output-dir', default='output/results/enhanced-ground-truth')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    csv_paths = sorted(repo_root.glob(args.input_glob))
    if not csv_paths:
        raise FileNotFoundError(f'No input files found with pattern: {args.input_glob}')

    config_path = (repo_root / args.config).resolve()
    engine = StabilityEngine(str(config_path))

    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    generated = 0
    for csv_path in csv_paths:
        out_path = out_dir / csv_path.name.replace('.csv', '_enhanced_ground_truth.csv')
        if out_path.exists() and not args.overwrite:
            print(f'Skipping existing file: {out_path}')
            continue

        df = pd.read_csv(csv_path)
        gt_df = build_enhanced_ground_truth(df, engine)
        gt_df.to_csv(out_path, index=False)

        total_rows += len(gt_df)
        generated += 1
        print(f'Generated: {out_path} ({len(gt_df)} rows)')

    print(f'Completed enhanced ground truth generation: files={generated}, rows={total_rows}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Visualize model performance metrics across devices and algorithms.
Generates comparison charts for R², MAE, accuracy, generalization gap, and stability.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def find_repo_root() -> Path:
    """Find repository root by looking for pyproject.toml"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def load_model_metrics(model_dir: Path, model_name: str = None) -> Dict:
    """Cargar métricas desde un directorio de modelo - STRICTO"""
    history_file = None
    
    # Busca ESPECÍFICAMENTE el archivo para el modelo solicitado
    if model_name:
        history_file = model_dir / f"adaptive_w_model_{model_name}_history.json"
        if not history_file.exists():
            return None  # ← SI NO EXISTE, RETORNA NONE (no tomar otro)
    
    if not history_file:
        return None
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        best = data.get('best', {})
        return {
            'trial': best.get('trial', 0),
            'holdout_r2': best.get('holdout_r2', 0),
            'cv_r2_mean': best.get('cv_r2_mean', 0),
            'cv_r2_std': best.get('cv_r2_std', 0),
            'cv_rmse_mean': best.get('cv_rmse_mean', 0),
            'cv_mae_mean': best.get('cv_mae_mean', 0),
            'holdout_rmse': best.get('holdout_rmse', 0),
            'holdout_mae': best.get('holdout_mae', 0),
            'generalization_gap': best.get('generalization_gap', 0),
        }
    except Exception as e:
        print(f"Error cargando {history_file}: {e}")
        return None


def aggregate_metrics(repo_root: Path) -> Dict[str, Dict]:
    """Aggregate all model metrics by device and algorithm"""
    
    devices = {
        '23': 'DOBACK023\n(K1=1.15)',
        '24': 'DOBACK024\n(K1=1.15)',
        '27': 'DOBACK027\n(K1=1.50 ⚠)',
        '28': 'DOBACK028\n(K1=1.15)'
    }
    
    models = ['rf', 'gbr', 'extra_trees']
    
    results = {}
    
    for device_id, device_label in devices.items():
        device_dir = repo_root / 'output' / 'models' / f'doback-{device_id}'
        
        if not device_dir.exists():
            print(f"⚠️  Directory not found: {device_dir}")
            continue
        
        # Check for model subdirectories
        model_dirs = list(device_dir.glob('*/'))
        if not model_dirs:
            # Models in same directory
            model_dirs = [device_dir]
        
        results[device_id] = {}
        
        for model in models:
            # Try multiple naming patterns
            possible_dirs = [
                device_dir / model,
                device_dir,
            ]
            
            metrics = None
            for possible_dir in possible_dirs:
                if possible_dir.exists():
                    # Pass model name for specific file search
                    metrics = load_model_metrics(possible_dir, model_name=model)
                    if metrics:
                        break
            
            if metrics:
                results[device_id][model] = metrics
            else:
                print(f"⚠️  No data found for DOBACK{device_id} - {model}")
    
    return results, devices, models


def create_comparison_dataframe(results: Dict, devices: Dict, models: List) -> pd.DataFrame:
    """Convert results to DataFrame for easier plotting"""
    
    rows = []
    for device_id, device_label in devices.items():
        if device_id not in results:
            continue
        
        for model in models:
            if model not in results[device_id]:
                continue
            
            metrics = results[device_id][model]
            rows.append({
                'Device': device_id,
                'Device Label': device_label,
                'Model': model,
                'Model Type': model.upper() if model != 'gbr' else 'GBR',
                'R² (Holdout)': metrics['holdout_r2'],
                'R² (CV μ)': metrics['cv_r2_mean'],
                'R² (CV σ)': metrics['cv_r2_std'],
                'RMSE (Holdout)': metrics['holdout_rmse'],
                'MAE (Holdout)': metrics['holdout_mae'],
                'RMSE (CV)': metrics['cv_rmse_mean'],
                'MAE (CV)': metrics['cv_mae_mean'],
                'Generalization Gap': metrics['generalization_gap'],
            })
    
    return pd.DataFrame(rows)


def plot_r2_comparison(df: pd.DataFrame):
    """Build R² comparison figure across devices and models."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Holdout R²
    pivot_holdout = df.pivot_table(
        values='R² (Holdout)', 
        index='Device Label', 
        columns='Model Type'
    )
    pivot_holdout.plot(kind='bar', ax=axes[0], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0].set_title('R² Score (Holdout Set)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_xlabel('Device', fontsize=12)
    axes[0].axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target R²=0.70', alpha=0.7)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # CV Mean R²
    pivot_cv = df.pivot_table(
        values='R² (CV μ)', 
        index='Device Label', 
        columns='Model Type'
    )
    pivot_cv.plot(kind='bar', ax=axes[1], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1].set_title('R² Score (Cross-Validation Mean)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_xlabel('Device', fontsize=12)
    axes[1].axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target R²=0.70', alpha=0.7)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_error_metrics(df: pd.DataFrame):
    """Build RMSE and MAE comparison figure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE
    pivot_rmse = df.pivot_table(
        values='RMSE (Holdout)', 
        index='Device Label', 
        columns='Model Type'
    )
    pivot_rmse.plot(kind='bar', ax=axes[0], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0].set_title('RMSE (Holdout Set) - Lower is Better', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_xlabel('Device', fontsize=12)
    axes[0].legend(title='Model')
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # MAE
    pivot_mae = df.pivot_table(
        values='MAE (Holdout)', 
        index='Device Label', 
        columns='Model Type'
    )
    pivot_mae.plot(kind='bar', ax=axes[1], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1].set_title('MAE (Holdout Set) - Lower is Better', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE (deg/s)', fontsize=12)
    axes[1].set_xlabel('Device', fontsize=12)
    axes[1].legend(title='Model')
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_generalization_gap(df: pd.DataFrame):
    """Build generalization gap figure (overfitting indicator)."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_gap = df.pivot_table(
        values='Generalization Gap', 
        index='Device Label', 
        columns='Model Type'
    )
    
    pivot_gap.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    ax.set_title('Generalization Gap (Overfitting Indicator) - Lower is Better', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Gap (Holdout R² - CV R² mean)', fontsize=12)
    ax.set_xlabel('Device', fontsize=12)
    ax.axhline(y=0.08, color='red', linestyle='--', linewidth=2, label='Max threshold=0.08', alpha=0.7)
    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_cv_stability(df: pd.DataFrame):
    """Build CV R² standard deviation figure (model stability)."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_std = df.pivot_table(
        values='R² (CV σ)', 
        index='Device Label', 
        columns='Model Type'
    )
    
    pivot_std.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    ax.set_title('CV R² Std Deviation (Stability) - Lower is Better', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Std Dev', fontsize=12)
    ax.set_xlabel('Device', fontsize=12)
    ax.axhline(y=0.08, color='red', linestyle='--', linewidth=2, label='Max threshold=0.08', alpha=0.7)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame):
    """Build heatmap figure for all key metrics."""
    
    # Prepare data for heatmap
    metrics_to_plot = ['R² (Holdout)', 'RMSE (Holdout)', 'MAE (Holdout)', 'Generalization Gap', 'R² (CV σ)']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        pivot = df.pivot_table(
            values=metric, 
            index='Device Label', 
            columns='Model Type'
        )
        
        # Use different decimal precision based on metric type
        if 'RMSE' in metric or 'MAE' in metric:
            fmt = '.0f'  # No decimals for large error values
        else:
            fmt = '.2f'  # 2 decimals for R² and Gap (smaller values)
        
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt=fmt,
            cmap='RdYlGn' if 'Gap' not in metric and 'σ' not in metric and 'RMSE' not in metric and 'MAE' not in metric else 'RdYlGn_r',
            ax=axes[idx],
            cbar_kws={'label': metric}
        )
        axes[idx].set_title(metric, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_model_ranking(df: pd.DataFrame):
    """Build overall model ranking figure."""
    
    # Normalize metrics for scoring (R² higher is better, others lower is better)
    df_temp = df.copy()
    df_temp['Score'] = (
        (df_temp['R² (Holdout)'] / df_temp['R² (Holdout)'].max()) * 0.4 +  # 40% weight
        (1 - df_temp['RMSE (Holdout)'] / df_temp['RMSE (Holdout)'].max()) * 0.3 +  # 30% weight
        (1 - df_temp['Generalization Gap'] / df_temp['Generalization Gap'].max()) * 0.2 +  # 20% weight
        (1 - df_temp['R² (CV σ)'] / df_temp['R² (CV σ)'].max()) * 0.1  # 10% weight
    )
    
    df_temp['Device-Model'] = df_temp['Device'] + '-' + df_temp['Model'].str.upper()
    df_temp = df_temp.sort_values('Score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB' if m == 'RF' else '#A23B72' if m == 'GBR' else '#F18F01' 
              for m in df_temp['Model'].str.upper()]
    
    bars = ax.barh(df_temp['Device-Model'], df_temp['Score'], color=colors)
    
    ax.set_title('Overall Model Performance Ranking', fontsize=14, fontweight='bold')
    ax.set_xlabel('Composite Score (R²: 40%, RMSE: 30%, Gap: 20%, Stability: 10%)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, df_temp['Score'])):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_summary_report(df: pd.DataFrame, output_dir: Path):
    """Create a text summary report"""
    
    report = []
    report.append("=" * 80)
    report.append("MODEL PERFORMANCE SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Best models per metric
    report.append("🏆 BEST MODELS BY METRIC")
    report.append("-" * 80)
    
    best_holdout_r2 = df.loc[df['R² (Holdout)'].idxmax()]
    report.append(f"\nBest R² (Holdout): DOBACK{best_holdout_r2['Device']}-{best_holdout_r2['Model'].upper()}")
    report.append(f"  R²: {best_holdout_r2['R² (Holdout)']:.4f}")
    report.append(f"  RMSE: {best_holdout_r2['RMSE (Holdout)']:.4f}")
    report.append(f"  GAP: {best_holdout_r2['Generalization Gap']:.4f}")
    
    best_rmse = df.loc[df['RMSE (Holdout)'].idxmin()]
    report.append(f"\nBest RMSE: DOBACK{best_rmse['Device']}-{best_rmse['Model'].upper()}")
    report.append(f"  R²: {best_rmse['R² (Holdout)']:.4f}")
    report.append(f"  RMSE: {best_rmse['RMSE (Holdout)']:.4f}")
    
    best_gap = df.loc[df['Generalization Gap'].idxmin()]
    report.append(f"\nBest Generalization: DOBACK{best_gap['Device']}-{best_gap['Model'].upper()}")
    report.append(f"  R²: {best_gap['R² (Holdout)']:.4f}")
    report.append(f"  GAP: {best_gap['Generalization Gap']:.4f}")
    
    # Device comparison
    report.append("\n" + "=" * 80)
    report.append("DEVICE COMPARISON")
    report.append("-" * 80)
    
    for device in sorted(df['Device'].unique()):
        device_data = df[df['Device'] == device]
        report.append(f"\nDOBACK{device}:")
        report.append(f"  Avg R²: {device_data['R² (Holdout)'].mean():.4f}")
        report.append(f"  Avg RMSE: {device_data['RMSE (Holdout)'].mean():.4f}")
        report.append(f"  Models: {len(device_data)}")
    
    # Target attainment
    report.append("\n" + "=" * 80)
    report.append("TARGET ATTAINMENT (R² ≥ 0.70)")
    report.append("-" * 80)
    
    on_target = df[df['R² (Holdout)'] >= 0.70]
    report.append(f"\nModels on target: {len(on_target)} / {len(df)}")
    
    if len(on_target) > 0:
        report.append("\nModels achieving R² ≥ 0.70:")
        for _, row in on_target.iterrows():
            report.append(f"  • DOBACK{row['Device']}-{row['Model'].upper()}: R²={row['R² (Holdout)']:.4f}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    filepath = output_dir / 'summary_report.txt'
    filepath.write_text(report_text)
    print(f"✓ Saved: {filepath}")
    
    return report_text


def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE VISUALIZATION SCRIPT")
    print("=" * 80 + "\n")
    
    repo_root = find_repo_root()
    print(f"📂 Repository root: {repo_root}\n")
    
    output_dir = repo_root / 'output' / 'leaderboard'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Aggregate metrics
    print("📊 Aggregating metrics...\n")
    results, devices, models = aggregate_metrics(repo_root)
    
    if not results:
        print("❌ No model data found in output/models/")
        print("   Make sure you've run adaptive_hyperparam_search.py for all devices")
        return 1
    
    # Create comparison dataframe
    df = create_comparison_dataframe(results, devices, models)
    print(f"✓ Loaded {len(df)} model results\n")
    print(df.to_string())
    print()
    
    # Create visualizations
    print("\n📈 Generating visualizations...\n")
    compact_pdf_path = output_dir / 'models_visualizations_compact.pdf'
    
    try:
        with PdfPages(compact_pdf_path) as pdf:
            for build_plot in [
                plot_r2_comparison,
                plot_error_metrics,
                plot_generalization_gap,
                plot_cv_stability,
                plot_heatmap,
                plot_model_ranking,
            ]:
                fig = build_plot(df)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f"✓ Saved compact plots: {compact_pdf_path}")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create summary report
    print("\n📄 Generating summary report...\n")
    report_text = create_summary_report(df, output_dir)
    print(report_text)
    
    print("=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

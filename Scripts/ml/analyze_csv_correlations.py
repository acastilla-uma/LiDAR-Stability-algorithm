"""
Quick Correlation Analysis for Pre-computed Data

Analyzes correlations between LiDAR features (φ_lidar, TRI, ruggedness) 
and Stability Index that are already present in CSV files.

No feature extraction needed - just correlation analysis and visualization.

Usage:
    python Scripts/ml/analyze_csv_correlations.py \
        --file Doback-Data/map-matched/DOBACK024_20251009_seg87.csv \
        --output output/correlations_report.html
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]


class CSVCorrelationAnalyzer:
    """Analyzes correlations in pre-computed CSV data."""
    
    def __init__(self):
        self.df = None
        self.correlations = {}
    
    def load_csv(self, csv_path: Path, sample_rate: int = 1) -> bool:
        """
        Load CSV file.
        
        Args:
            csv_path: Path to CSV
            sample_rate: Process every Nth row
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading: {csv_path.name}")
            df = pd.read_csv(csv_path)
            
            # Apply sampling
            if sample_rate > 1:
                df = df.iloc[::sample_rate].copy()
                logger.info(f"  Applied sampling rate {sample_rate} → {len(df)} rows")
            
            logger.info(f"  Total rows: {len(df)}")
            logger.info(f"  Columns: {', '.join(df.columns.tolist())}")
            
            self.df = df
            return True
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False
    
    def compute_correlations(self) -> Dict:
        """
        Compute Pearson and Spearman correlations between LiDAR features and SI.
        
        Returns:
            Dict with correlation results
        """
        if self.df is None:
            logger.error("No data loaded")
            return {}
        
        # Define features to analyze
        features = ['phi_lidar', 'tri', 'ruggedness']
        target = 'si'
        
        # Check if target exists
        if target not in self.df.columns:
            logger.error(f"Target column '{target}' not found in CSV")
            logger.info(f"Available columns: {self.df.columns.tolist()}")
            return {}
        
        correlations = {}
        
        for feat in features:
            # Skip if feature not in CSV
            if feat not in self.df.columns:
                logger.warning(f"Feature '{feat}' not found in CSV (skipping)")
                continue
            
            # Get valid pairs (no NaN)
            valid_mask = self.df[feat].notna() & self.df[target].notna()
            feat_vals = self.df.loc[valid_mask, feat].values
            target_vals = self.df.loc[valid_mask, target].values
            
            n_valid = len(feat_vals)
            logger.info(f"\n{feat}: {n_valid} valid pairs")
            
            if n_valid < 3:
                logger.warning(f"  Too few data points for {feat}")
                correlations[feat] = {
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'spearman_r': np.nan,
                    'spearman_p': np.nan,
                    'n_valid': n_valid,
                    'slope': np.nan,
                    'intercept': np.nan
                }
                continue
            
            # Compute correlations
            try:
                pearson_r, pearson_p = pearsonr(feat_vals, target_vals)
                spearman_r, spearman_p = spearmanr(feat_vals, target_vals)
                slope, intercept, r_value, p_value, std_err = linregress(feat_vals, target_vals)
            except Exception as e:
                logger.error(f"  Correlation failed: {e}")
                pearson_r = pearson_p = spearman_r = spearman_p = slope = intercept = np.nan
            
            correlations[feat] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'n_valid': n_valid,
                'slope': float(slope),
                'intercept': float(intercept)
            }
            
            # Log results
            logger.info(f"  Pearson r:  {pearson_r:7.3f} (p={pearson_p:.4f})")
            logger.info(f"  Spearman ρ: {spearman_r:7.3f} (p={spearman_p:.4f})")
            logger.info(f"  Linear fit: y = {slope:.6f}*x + {intercept:.3f}")
        
        self.correlations = correlations
        return correlations
    
    def generate_html_report(self, output_path: Path) -> None:
        """
        Generate interactive HTML report with scatter plots.
        
        Args:
            output_path: Output HTML file path
        """
        if self.df is None or not self.correlations:
            logger.error("No data or correlations to plot")
            return
        
        logger.info(f"\nGenerating HTML report...")
        
        features_info = {
            'phi_lidar': {
                'label': 'φ_lidar (Transverse Slope)',
                'unit': '[rad]',
                'color': 'rgb(31, 119, 180)'
            },
            'tri': {
                'label': 'TRI (Terrain Roughness Index)',
                'unit': '[m]',
                'color': 'rgb(255, 127, 14)'
            },
            'ruggedness': {
                'label': 'Ruggedness Index',
                'unit': '[m]',
                'color': 'rgb(44, 160, 44)'
            }
        }
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"{features_info.get(f, {}).get('label', f)}" 
                for f in ['phi_lidar', 'tri', 'ruggedness']
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        col_idx = 1
        for feat_key in ['phi_lidar', 'tri', 'ruggedness']:
            if feat_key not in self.df.columns:
                col_idx += 1
                continue
            
            feat_info = features_info.get(feat_key, {})
            
            # Get valid data
            valid_mask = self.df[feat_key].notna() & self.df['si'].notna()
            x_vals = self.df.loc[valid_mask, feat_key].values.astype(float)
            y_vals = self.df.loc[valid_mask, 'si'].values.astype(float)
            
            if len(x_vals) < 2:
                col_idx += 1
                continue
            
            logger.info(f"  Plotting {feat_key}: {len(x_vals)} points, "
                       f"X=[{x_vals.min():.4f}, {x_vals.max():.4f}], "
                       f"Y=[{y_vals.min():.4f}, {y_vals.max():.4f}]")
            
            # Convert to Python lists for proper JSON serialization
            x_list = x_vals.tolist()
            y_list = y_vals.tolist()
            
            # Scatter plot
            fig.add_trace(
                go.Scattergl(
                    x=x_list, y=y_list,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=feat_info.get('color', 'blue'),
                        opacity=0.5,
                    ),
                    name=feat_key,
                    hovertemplate=f'<b>{feat_key}</b><br>' +
                                 f'X: %{{x:.4f}}<br>' +
                                 f'SI: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=col_idx
            )
            
            # Add regression line
            if feat_key in self.correlations:
                corr = self.correlations[feat_key]
                slope = corr['slope']
                intercept = corr['intercept']
                
                if not np.isnan(slope):
                    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
                    y_line = slope * x_line + intercept
                    
                    # Annotation with correlation info
                    pearson_r = corr['pearson_r']
                    spearman_r = corr['spearman_r']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_line.tolist(), y=y_line.tolist(),
                            mode='lines',
                            line=dict(
                                color='red',
                                width=3,
                            ),
                            name=f'r={pearson_r:.3f}, ρ={spearman_r:.3f}',
                            showlegend=True,
                            hoverinfo='skip'
                        ),
                        row=1, col=col_idx
                    )
            
            # Set explicit axis ranges with small padding
            x_pad = (x_vals.max() - x_vals.min()) * 0.05 or 0.01
            y_pad = (y_vals.max() - y_vals.min()) * 0.05 or 0.01
            
            fig.update_xaxes(
                title_text=f"{feat_info.get('label', feat_key)} {feat_info.get('unit', '')}",
                range=[float(x_vals.min() - x_pad), float(x_vals.max() + x_pad)],
                row=1, col=col_idx
            )
            fig.update_yaxes(
                title_text="SI (Stability Index)",
                range=[float(y_vals.min() - y_pad), float(y_vals.max() + y_pad)],
                row=1, col=col_idx
            )
            
            col_idx += 1
        
        # Update layout
        fig.update_layout(
            title_text="LiDAR Features vs Stability Index Correlations",
            title_font_size=18,
            height=550,
            width=1400,
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
        )
        
        # Create statistics table
        stats_html = self._create_stats_table()
        
        # Create data summary
        summary_stats = self._create_data_summary()
        
        # Generate plotly div (not full HTML, with CDN reference)
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Combine into final HTML
        final_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LiDAR-SI Correlation Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1500px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .plot-section {{
            background-color: #fafafa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        .plot-section h2 {{
            margin-bottom: 15px;
            color: #333;
            font-size: 18px;
        }}
        .stats-section {{
            margin-bottom: 30px;
        }}
        .stats-section h2 {{
            margin-bottom: 15px;
            color: #333;
            font-size: 18px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        .metric-label {{
            font-weight: 600;
            color: #333;
        }}
        .positive {{
            color: #00b894;
            font-weight: 600;
        }}
        .negative {{
            color: #d63031;
            font-weight: 600;
        }}
        .weak {{
            color: #636e72;
        }}
        .significant {{
            background-color: #fff3cd;
        }}
        .not-significant {{
            background-color: #f8f9fa;
        }}
        .footnote {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f5f7fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            font-size: 13px;
            color: #555;
            line-height: 1.6;
        }}
        .footnote h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .footnote ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        .footnote li {{
            margin-bottom: 8px;
        }}
        .code {{
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 10px;
        }}
        .badge-significant {{
            background-color: #00b894;
            color: white;
        }}
        .badge-weak {{
            background-color: #d3d3d3;
            color: #333;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        @media (max-width: 1024px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗗 LiDAR Terrain Features vs Stability Index</h1>
            <p>Correlation Analysis Report</p>
            <p style="font-size: 12px; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="plot-section">
                <h2>Interactive Scatter Plots with Linear Regression</h2>
                {plot_div}
            </div>
            
            <div class="stats-section">
                <h2>📊 Correlation Statistics</h2>
                {stats_html}
            </div>
            
            <div class="stats-section">
                <h2>📈 Data Summary</h2>
                {summary_stats}
            </div>
            
            <div class="footnote">
                <h3>📝 Interpretation Guide</h3>
                <ul>
                    <li><strong>φ_lidar:</strong> Transverse topographic slope (rad). Higher = steeper cross-slope terrain.</li>
                    <li><strong>TRI:</strong> Terrain Roughness Index (m). Higher = rougher terrain (more elevation variability).</li>
                    <li><strong>Ruggedness:</strong> Mean absolute elevation differences (m). Alternative roughness metric.</li>
                    <li><strong>SI:</strong> Vehicle Stability Index (0-1). Lower = more stable (safer).</li>
                    <li><strong>Pearson r:</strong> Linear correlation (-1 to 1). Measures strength of linear relationship.</li>
                    <li><strong>Spearman ρ:</strong> Rank correlation. Non-parametric, robust to outliers.</li>
                    <li><strong>p-value:</strong> Statistical significance. p &lt; 0.05 = significant relationship.</li>
                    <li><strong>Slope:</strong> Linear regression coefficient. ΔSI per unit of terrain feature.</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        logger.info(f"✓ Report saved: {output_path}")
    
    def _create_stats_table(self) -> str:
        """Create HTML table with correlation statistics."""
        rows = []
        
        for feat_key, feat_name in [
            ('phi_lidar', 'φ_lidar (Transverse Slope)'),
            ('tri', 'TRI (Roughness Index)'),
            ('ruggedness', 'Ruggedness Index')
        ]:
            if feat_key not in self.correlations:
                continue
            
            corr = self.correlations[feat_key]
            n = corr['n_valid']
            
            if n < 3:
                rows.append(f"""
                <tr>
                    <td class="metric-label">{feat_name}</td>
                    <td colspan="6" style="color: #999;">Insufficient data (n={n})</td>
                </tr>
                """)
                continue
            
            # Determine significance
            pearson_sig = corr['pearson_p'] < 0.05
            spearman_sig = corr['spearman_p'] < 0.05
            
            pearson_class = 'positive' if abs(corr['pearson_r']) > 0.3 else 'weak'
            spearman_class = 'positive' if abs(corr['spearman_r']) > 0.3 else 'weak'
            
            rows.append(f"""
            <tr class="{'significant' if pearson_sig or spearman_sig else 'not-significant'}">
                <td class="metric-label">{feat_name}</td>
                <td class="{pearson_class}">{corr['pearson_r']:7.3f}</td>
                <td>{corr['pearson_p']:.4e}</td>
                <td><span class="badge {'badge-significant' if pearson_sig else 'badge-weak'}">{'✓' if pearson_sig else '✗'}</span></td>
                <td class="{spearman_class}">{corr['spearman_r']:7.3f}</td>
                <td>{corr['spearman_p']:.4e}</td>
                <td><span class="badge {'badge-significant' if spearman_sig else 'badge-weak'}">{'✓' if spearman_sig else '✗'}</span></td>
                <td>{n}</td>
            </tr>
            """)
        
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>Terrain Feature</th>
                    <th>Pearson r</th>
                    <th>p-value</th>
                    <th>Sig.</th>
                    <th>Spearman ρ</th>
                    <th>p-value</th>
                    <th>Sig.</th>
                    <th>N</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">
            <strong>Note:</strong> Sig. = Statistically significant (p &lt; 0.05). ✓ = Significant | ✗ = Not significant
        </p>
        """
        
        return table_html
    
    def _create_data_summary(self) -> str:
        """Create HTML table with data summary statistics."""
        if self.df is None:
            return "<p>No data available</p>"
        
        features = ['phi_lidar', 'tri', 'ruggedness', 'si']
        rows = []
        
        for feat in features:
            if feat not in self.df.columns:
                continue
            
            data = self.df[feat].dropna()
            
            if len(data) == 0:
                rows.append(f"""
                <tr>
                    <td class="metric-label">{feat}</td>
                    <td colspan="5" style="color: #999;">All values are NaN</td>
                </tr>
                """)
                continue
            
            rows.append(f"""
            <tr>
                <td class="metric-label">{feat}</td>
                <td>{len(data)}</td>
                <td>{data.mean():.6f}</td>
                <td>{data.std():.6f}</td>
                <td>{data.min():.6f}</td>
                <td>{data.max():.6f}</td>
            </tr>
            """)
        
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
        
        return table_html


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations in pre-computed CSV data"
    )
    parser.add_argument('--file', type=str, required=True,
                       help='CSV file to analyze (absolute or relative path)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file (default: output/correlation_<timestamp>.html)')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth row (default: 1)')
    
    args = parser.parse_args()
    
    # Resolve file path
    csv_path = Path(args.file)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = CSVCorrelationAnalyzer()
    
    # Load data
    if not analyzer.load_csv(csv_path, args.sample_rate):
        sys.exit(1)
    
    # Compute correlations
    logger.info("\n" + "="*60)
    logger.info("COMPUTING CORRELATIONS")
    logger.info("="*60)
    
    correlations = analyzer.compute_correlations()
    
    if not correlations:
        logger.error("No correlations computed")
        sys.exit(1)
    
    # Generate report
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / f"correlation_{timestamp}.html"
    
    analyzer.generate_html_report(output_path)
    
    logger.info("\n✓ Analysis complete!")
    logger.info(f"  Input file: {csv_path}")
    logger.info(f"  Report: {output_path}")


if __name__ == "__main__":
    main()

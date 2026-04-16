#!/usr/bin/env python3
"""
Compute physics-based stability index from a CSV and compare against measured SI.

Usage:
    python src/lidar_stability/physics/compare_stability_csv.py --csv path/to/file.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.physics import StabilityEngine


ROLL_CANDIDATES = ["roll_deg", "roll", "phi_roll_deg", "roll_angle_deg"]
AX_CANDIDATES = ["ax", "acc_x", "accel_x"]
AZ_CANDIDATES = ["az", "acc_z", "accel_z"]
GY_CANDIDATES = ["gy", "gyro_y", "gyro_y_deg_s", "wy_deg_s"]
OMEGA_CANDIDATES = [
    "gz",
    "gy",
    "gx",
    "gyro_z_deg_s",
    "gyro_y_deg_s",
    "gyro_x_deg_s",
    "yaw_rate_deg_s",
    "omega_deg_s",
    "roll_rate_deg_s",
    "omega",
    "roll_rate",
    "yaw",
    "yaw_deg",
    "wx_deg_s",
]
SI_CANDIDATES = ["si", "si_mcu", "si_real", "stability_index", "stability"]
TIME_CANDIDATES = ["timestamp", "time", "t_us", "timeantwifi"]


def find_column(df: pd.DataFrame, candidates: list[str], forced: Optional[str], label: str) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(
                f"La columna forzada para {label} no existe: '{forced}'. "
                f"Disponibles: {df.columns.tolist()}"
            )
        return forced

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"No se encontrÃ³ columna para {label}. "
        f"Candidatas: {candidates}. Disponibles: {df.columns.tolist()}"
    )


def _time_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a time-like column to elapsed seconds."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.to_numpy(dtype=float)
        vals = vals - vals[0]
        if np.nanmax(np.abs(vals)) > 1e5:
            vals = vals / 1e6
        return vals

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() < max(3, len(series) // 2):
        raise ValueError(
            "No se pudo convertir la columna temporal a segundos para derivar omega desde yaw."
        )

    vals = ((dt - dt.iloc[0]).dt.total_seconds()).to_numpy(dtype=float)
    return vals


def _omega_from_yaw_deg(yaw_deg: np.ndarray, time_seconds: np.ndarray) -> np.ndarray:
    """Estimate angular rate (deg/s) from yaw angle (deg) using dÏˆ/dt."""
    yaw_rad = np.unwrap(np.radians(np.asarray(yaw_deg, dtype=float)))
    t = np.asarray(time_seconds, dtype=float)

    if len(yaw_rad) < 2:
        return np.zeros_like(yaw_rad)

    dt = np.diff(t)
    bad = ~np.isfinite(dt) | (np.abs(dt) < 1e-9)
    if np.any(bad):
        t = t.copy()
        for i in np.where(bad)[0]:
            t[i + 1] = t[i] + 1e-3

    omega_rad_s = np.gradient(yaw_rad, t)
    return np.degrees(omega_rad_s)


def _infer_yaw_scale(yaw_values: np.ndarray) -> float:
    """Infer yaw scale to convert raw yaw units to degrees."""
    vals = np.asarray(yaw_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0

    p99 = float(np.percentile(np.abs(vals), 99))
    if p99 > 720:
        return 0.01
    return 1.0


def _infer_omega_scale(col_name: str, omega_values: np.ndarray) -> float:
    """Infer omega scale to convert raw omega units to deg/s."""
    name = (col_name or "").lower()
    if name in {"gx", "gy", "gz"}:
        return 0.001

    vals = np.asarray(omega_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0

    p99 = float(np.percentile(np.abs(vals), 99))
    if p99 > 1000:
        return 0.001
    return 1.0


def _infer_gy_scale_to_mdeg(col_name: str) -> float:
    """Infer multiplicative scale to convert gy column to mdeg/s (firmware units)."""
    name = (col_name or "").lower()
    if name in {"gx", "gy", "gz", "gyro_x", "gyro_y", "gyro_z"}:
        return 1.0
    if "deg" in name:
        return 1000.0
    return 1.0


def _moving_average_ring(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with circular buffer behavior similar to firmware loop."""
    window = max(1, int(window))
    values = np.asarray(values, dtype=float)
    if window == 1 or len(values) == 0:
        return values.copy()

    buffer = np.zeros(window, dtype=float)
    out = np.zeros_like(values, dtype=float)
    index = 0
    for i, value in enumerate(values):
        buffer[index] = value
        index = (index + 1) % window
        out[i] = np.mean(buffer)
    return out


def _fit_params(phi_term: np.ndarray, omega_term: np.ndarray, si_real: np.ndarray, fit_bias: bool) -> tuple[float, float, float]:
    """Fit SI model parameters from data.

    Model:
      SI = c0 - k1*phi_term - k2*omega_term
    with c0 fixed to 1.0 unless fit_bias=True.
    """
    phi_term = np.asarray(phi_term, dtype=float)
    omega_term = np.asarray(omega_term, dtype=float)
    si_real = np.asarray(si_real, dtype=float)

    if fit_bias:
        X = np.column_stack([phi_term, omega_term, np.ones_like(phi_term)])
        beta, *_ = np.linalg.lstsq(X, si_real, rcond=None)
        k1 = max(0.0, float(-beta[0]))
        k2 = max(0.0, float(-beta[1]))
        c0 = float(beta[2])
        return k1, k2, c0

    y = 1.0 - si_real
    X = np.column_stack([phi_term, omega_term])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k1 = max(0.0, float(beta[0]))
    k2 = max(0.0, float(beta[1]))
    c0 = 1.0
    return k1, k2, c0


def _auto_select_omega_col(df: pd.DataFrame, si_col: str) -> Optional[str]:
    """Choose best omega column automatically, preferring direct gyro over yaw."""
    direct_candidates = [
        "gz", "gy", "gx",
        "gyro_z_deg_s", "gyro_y_deg_s", "gyro_x_deg_s",
        "yaw_rate_deg_s", "omega_deg_s", "roll_rate_deg_s", "omega", "roll_rate", "wx_deg_s",
    ]
    available = [col for col in direct_candidates if col in df.columns]
    if not available:
        return "yaw" if "yaw" in df.columns else ("yaw_deg" if "yaw_deg" in df.columns else None)

    si = df[si_col].astype(float)
    best_col = available[0]
    best_score = -np.inf

    for col in available:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scale = _infer_omega_scale(col, vals)
        vals = vals * scale
        score = np.corrcoef(si.to_numpy(dtype=float), np.abs(vals))[0, 1]
        score = float(np.nan_to_num(np.abs(score), nan=0.0))
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _corr_with_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Correlation between a and b with lag applied to b.

    Positive lag means b is shifted forward (b happens later).
    """
    if lag > 0:
        a2 = a[:-lag]
        b2 = b[lag:]
    elif lag < 0:
        a2 = a[-lag:]
        b2 = b[:lag]
    else:
        a2 = a
        b2 = b

    if len(a2) < 3:
        return np.nan
    return float(np.corrcoef(a2, b2)[0, 1])


def _mae_with_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """MAE between a and b with lag applied to b."""
    if lag > 0:
        a2 = a[:-lag]
        b2 = b[lag:]
    elif lag < 0:
        a2 = a[-lag:]
        b2 = b[:lag]
    else:
        a2 = a
        b2 = b

    if len(a2) < 3:
        return np.nan
    return float(np.mean(np.abs(a2 - b2)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcula SI con ecuaciÃ³n fÃ­sica y lo compara con SI del CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--mode",
        choices=["auto", "ino", "roll"],
        default="auto",
        help="Modo de cÃ¡lculo: firmware (.ino) o roll/omega",
    )
    parser.add_argument("--ax-col", default=None, help="Columna ax para modo ino")
    parser.add_argument("--az-col", default=None, help="Columna az para modo ino")
    parser.add_argument("--gy-col", default=None, help="Columna gy para modo ino (mdeg/s o deg/s)")
    parser.add_argument("--roll-col", default=None, help="Columna de roll en grados")
    parser.add_argument("--omega-col", default=None, help="Columna de velocidad angular en deg/s")
    parser.add_argument(
        "--omega-from-yaw",
        action="store_true",
        help="Interpretar la columna omega como yaw (Ã¡ngulo) y derivar d(yaw)/dt en deg/s",
    )
    parser.add_argument(
        "--yaw-scale",
        type=float,
        default=None,
        help="Escala multiplicativa de yaw para convertir a grados (auto si no se define)",
    )
    parser.add_argument(
        "--omega-scale",
        type=float,
        default=None,
        help="Escala multiplicativa de omega para convertir a deg/s (auto si no se define)",
    )
    parser.add_argument(
        "--max-omega-deg-s",
        type=float,
        default=120.0,
        help="Clipping absoluto para omega en deg/s tras conversiÃ³n",
    )
    parser.add_argument(
        "--fit-k",
        action="store_true",
        help="Ajustar k1 y k2 por mÃ­nimos cuadrados antes de comparar",
    )
    parser.add_argument(
        "--fit-bias",
        action="store_true",
        help="Permitir sesgo c0 en SI = c0 - k1*phi - k2*omega^2 (requiere --fit-k)",
    )
    parser.add_argument("--si-col", default=None, help="Columna de SI medido en CSV")
    parser.add_argument("--time-col", default=None, help="Columna temporal para eje X")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "lidar_stability" / "config" / "vehicle.yaml"),
        help="Ruta a vehicle.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "output" / "results"),
        help="Carpeta de salida",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Guardar CSV con columnas calculadas",
    )
    parser.add_argument(
        "--forensic",
        action="store_true",
        help="Exportar diagnÃ³stico detallado (tÃ©rminos intermedios y anÃ¡lisis de lag)",
    )
    parser.add_argument(
        "--forensic-csv",
        default=None,
        help="Ruta de salida para CSV forense (opcional)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=25,
        help="Lag mÃ¡ximo (muestras) para anÃ¡lisis temporal forense",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("El CSV estÃ¡ vacÃ­o")

    si_col = find_column(df, SI_CANDIDATES, args.si_col, "SI medido")

    use_ino_mode = False
    if args.mode == "ino":
        use_ino_mode = True
    elif args.mode == "auto":
        use_ino_mode = any(col in df.columns for col in AX_CANDIDATES) and any(col in df.columns for col in AZ_CANDIDATES)

    omega_col = None
    roll_col = None
    ax_col = None
    az_col = None
    gy_col = None

    if use_ino_mode:
        ax_col = find_column(df, AX_CANDIDATES, args.ax_col, "ax")
        az_col = find_column(df, AZ_CANDIDATES, args.az_col, "az")
        if args.gy_col:
            gy_col = find_column(df, GY_CANDIDATES, args.gy_col, "gy")
        else:
            for col in GY_CANDIDATES:
                if col in df.columns:
                    gy_col = col
                    break
    else:
        roll_col = find_column(df, ROLL_CANDIDATES, args.roll_col, "roll")
        if args.omega_col:
            omega_col = find_column(df, OMEGA_CANDIDATES, args.omega_col, "omega")
        else:
            omega_col = _auto_select_omega_col(df, si_col)

    time_col = None
    if args.time_col:
        if args.time_col not in df.columns:
            raise ValueError(f"La columna temporal no existe: {args.time_col}")
        time_col = args.time_col
    else:
        for col in TIME_CANDIDATES:
            if col in df.columns:
                time_col = col
                break

    engine = StabilityEngine(config_path=args.config)

    selected_cols = [si_col]
    if use_ino_mode:
        selected_cols.extend([ax_col, az_col])
        if gy_col:
            selected_cols.append(gy_col)
    else:
        selected_cols.append(roll_col)
        if omega_col:
            selected_cols.append(omega_col)
    if time_col:
        selected_cols.append(time_col)

    work_df = df[selected_cols].copy()
    if use_ino_mode:
        work_df = work_df.dropna(subset=[ax_col, az_col, si_col]).copy()
    else:
        work_df = work_df.dropna(subset=[roll_col, si_col]).copy()
    if work_df.empty:
        raise ValueError("No hay filas vÃ¡lidas tras eliminar NaN en roll/SI")

    omega_deg_s = None
    omega_source_desc = "no encontrada (Ï‰=0)"
    if (not use_ino_mode) and omega_col:
        use_yaw_derivative = args.omega_from_yaw or omega_col.lower() in {"yaw", "yaw_deg"}
        if use_yaw_derivative:
            if not time_col:
                raise ValueError(
                    "Para derivar omega desde yaw hace falta una columna temporal. "
                    "Pasa --time-col o incluye timestamp/timeantwifi en el CSV."
                )
            raw_yaw = work_df[omega_col].to_numpy(dtype=float)
            yaw_scale = args.yaw_scale if args.yaw_scale is not None else _infer_yaw_scale(raw_yaw)
            yaw_deg = raw_yaw * yaw_scale
            time_seconds = _time_to_seconds(work_df[time_col])
            omega_deg_s = _omega_from_yaw_deg(yaw_deg, time_seconds)
            omega_source_desc = f"derivada desde yaw ({omega_col}) con escala={yaw_scale}"
        else:
            raw_omega = work_df[omega_col].fillna(0.0).to_numpy(dtype=float)
            omega_scale = args.omega_scale if args.omega_scale is not None else _infer_omega_scale(omega_col, raw_omega)
            omega_deg_s = raw_omega * omega_scale
            omega_source_desc = f"directa desde columna {omega_col} con escala={omega_scale}"

        if args.max_omega_deg_s is not None and args.max_omega_deg_s > 0:
            omega_deg_s = np.clip(omega_deg_s, -args.max_omega_deg_s, args.max_omega_deg_s)
            omega_source_desc += f", clip=Â±{args.max_omega_deg_s} deg/s"

    if omega_deg_s is None:
        omega_deg_s = np.zeros(len(work_df), dtype=float)

    if use_ino_mode:
        gy_mdeg_s = np.zeros(len(work_df), dtype=float)
        if gy_col:
            raw_gy = work_df[gy_col].fillna(0.0).to_numpy(dtype=float)
            gy_scale = _infer_gy_scale_to_mdeg(gy_col)
            gy_mdeg_s = raw_gy * gy_scale
            omega_source_desc = f"firmware gy desde {gy_col} con escala={gy_scale} (mdeg/s)"
        else:
            omega_source_desc = "sin gy (tÃ©rmino dinÃ¡mico=0)"

        gy_avg = _moving_average_ring(gy_mdeg_s, engine.gy_avg_window)
        ax_vals = work_df[ax_col].to_numpy(dtype=float)
        az_vals = work_df[az_col].to_numpy(dtype=float)
        az_safe = np.where(np.abs(az_vals) < 1e-9, 1e-9, az_vals)

        phi_term = np.abs(np.arctan(ax_vals / az_safe)) / engine.phi_crit_ino_rad
        omega_term = (gy_avg / engine.wcrit_mdeg_s) ** 2

        work_df["phi_rad"] = np.abs(np.arctan(ax_vals / az_safe))
        work_df["phi_term"] = phi_term
        work_df["gy_raw_mdeg_s"] = gy_mdeg_s
        work_df["gy_avg_mdeg_s"] = gy_avg
        work_df["wcrit_mdeg_s"] = engine.wcrit_mdeg_s
        work_df["omega_term"] = omega_term
    else:
        roll_rad = np.radians(work_df[roll_col].to_numpy(dtype=float))
        omega_rad_s = np.radians(omega_deg_s)
        phi_term = np.abs(roll_rad) / engine.phi_c_rad
        omega_term = (np.abs(omega_rad_s) / engine.omega_crit_rad_s) ** 2

        work_df["roll_rad"] = roll_rad
        work_df["phi_term"] = phi_term
        work_df["omega_deg_s"] = omega_deg_s
        work_df["omega_rad_s"] = omega_rad_s
        work_df["omega_term"] = omega_term

    k1_used = float(engine.k1)
    k2_used = float(engine.k2)
    c0_used = 1.0
    if args.fit_k:
        k1_used, k2_used, c0_used = _fit_params(
            phi_term=phi_term,
            omega_term=omega_term,
            si_real=work_df[si_col].to_numpy(dtype=float),
            fit_bias=args.fit_bias,
        )

    work_df["si_calculada"] = c0_used - k1_used * phi_term - k2_used * omega_term
    work_df["error_abs"] = np.abs(work_df["si_calculada"] - work_df[si_col].to_numpy(dtype=float))
    work_df["error_signed"] = work_df["si_calculada"] - work_df[si_col].to_numpy(dtype=float)

    mae = float(work_df["error_abs"].mean())
    rmse = float(np.sqrt(np.mean((work_df["si_calculada"] - work_df[si_col]) ** 2)))
    corr = float(work_df[["si_calculada", si_col]].corr().iloc[0, 1])

    lag_values = list(range(-abs(args.max_lag), abs(args.max_lag) + 1))
    si_real_arr = work_df[si_col].to_numpy(dtype=float)
    si_calc_arr = work_df["si_calculada"].to_numpy(dtype=float)
    lag_corrs = [
        _corr_with_lag(si_real_arr, si_calc_arr, lag) for lag in lag_values
    ]
    lag_maes = [
        _mae_with_lag(si_real_arr, si_calc_arr, lag) for lag in lag_values
    ]

    corr_abs = np.abs(np.nan_to_num(np.array(lag_corrs), nan=-np.inf))
    best_corr_idx = int(np.argmax(corr_abs))
    best_corr_lag = int(lag_values[best_corr_idx])
    best_corr = float(lag_corrs[best_corr_idx])

    mae_arr = np.nan_to_num(np.array(lag_maes), nan=np.inf)
    best_mae_idx = int(np.argmin(mae_arr))
    best_mae_lag = int(lag_values[best_mae_idx])
    best_mae = float(lag_maes[best_mae_idx])

    x = work_df[time_col] if time_col else np.arange(len(work_df))
    x_label = time_col if time_col else "sample"

    stem = csv_path.stem
    png_plot_path = out_dir / f"{stem}_si_comparacion.png"
    html_plot_path = out_dir / f"{stem}_si_comparacion.html"

    plot_path = None
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(x, work_df[si_col], label=f"SI CSV ({si_col})", linewidth=1.3)
        plt.plot(x, work_df["si_calculada"], label="SI calculada (fÃ­sica)", linewidth=1.2)
        plt.xlabel(x_label)
        plt.ylabel("Stability Index (SI)")
        plt.title("ComparaciÃ³n de estabilidad: SI calculada vs SI en CSV")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_plot_path, dpi=180)
        plt.close()
        plot_path = png_plot_path
    except ModuleNotFoundError:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=work_df[si_col], mode="lines", name=f"SI CSV ({si_col})"))
            fig.add_trace(go.Scatter(x=x, y=work_df["si_calculada"], mode="lines", name="SI calculada (fÃ­sica)"))
            fig.update_layout(
                title="ComparaciÃ³n de estabilidad: SI calculada vs SI en CSV",
                xaxis_title=x_label,
                yaxis_title="Stability Index (SI)",
                template="plotly_white",
            )
            fig.write_html(str(html_plot_path), include_plotlyjs="cdn")
            plot_path = html_plot_path
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "No hay librerÃ­a de grÃ¡ficos disponible. Instala matplotlib o plotly."
            ) from exc

    if args.save_csv:
        csv_out = out_dir / f"{stem}_si_comparacion.csv"
        work_df.to_csv(csv_out, index=False)
        print(f"CSV de salida: {csv_out}")

    forensic_csv_path = None
    if args.forensic:
        if args.forensic_csv:
            forensic_csv_path = Path(args.forensic_csv)
        else:
            forensic_csv_path = out_dir / f"{stem}_si_forensic.csv"
        forensic_csv_path.parent.mkdir(parents=True, exist_ok=True)
        work_df.to_csv(forensic_csv_path, index=False)

    print("\n=== Resultado comparaciÃ³n SI ===")
    print(f"Archivo entrada: {csv_path}")
    print(f"Modo cÃ¡lculo: {'ino' if use_ino_mode else 'roll'}")
    if use_ino_mode:
        print(f"Columnas ino: ax={ax_col}, az={az_col}, gy={gy_col if gy_col else 'no encontrada'}")
    else:
        print(f"Columna roll: {roll_col}")
    print(f"Columna omega/yaw: {omega_col if omega_col else 'no encontrada'}")
    print(f"Fuente omega usada: {omega_source_desc}")
    print(f"Columna SI medida: {si_col}")
    print(f"Columna tiempo: {time_col if time_col else 'index'}")
    print(f"ParÃ¡metros usados: c0={c0_used:.6f}, k1={k1_used:.6f}, k2={k2_used:.6f}, omega_crit={engine.omega_crit_rad_s:.6f} rad/s")
    print(f"Ajuste automÃ¡tico: {'sÃ­' if args.fit_k else 'no'}")
    print(f"Muestras usadas: {len(work_df)}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Corr: {corr:.6f}")
    print(f"Mejor corr por lag: {best_corr:.6f} en lag={best_corr_lag}")
    print(f"Mejor MAE por lag:  {best_mae:.6f} en lag={best_mae_lag}")
    print(f"GrÃ¡fica: {plot_path}")
    if forensic_csv_path is not None:
        print(f"Forensic CSV: {forensic_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


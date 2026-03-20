#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compara SI del CSV con SI teórico (ecuación lateral de cabina_v2_4.ino)."
    )
    p.add_argument("--csv", type=Path, default=Path("Doback-Data/processed-data/DOBACK023_20251012.csv"), help="CSV de entrada.")
    p.add_argument("--out", type=Path, default=None, help="CSV enriquecido; por defecto output/results/stability_debug_<stem>.csv")
    p.add_argument("--k1", type=float, default=1.15, help="Parámetro k1")
    p.add_argument("--k2", type=float, default=2.05, help="Parámetro k2")
    p.add_argument("--d1", type=float, default=4.2, help="d1 en metros")
    p.add_argument("--s-mm", type=float, default=1100.0, help="S en milímetros")
    p.add_argument("--coeff", type=float, default=7.14, help="Coeficiente dinámico")
    p.add_argument("--alphav", type=float, default=64.0, help="Parámetro alphav")
    p.add_argument("--window", type=int, default=10, help="Ventana de promedio móvil de gy")
    p.add_argument("--plot", action="store_true", help="Genera PNG con si vs si_teórico y error absoluto")
    return p.parse_args()


def gy_ring_avg(values: np.ndarray, window: int) -> np.ndarray:
    w = max(int(window), 1)
    buf = np.zeros(w, dtype=float)
    out = np.empty_like(values, dtype=float)
    total = 0.0
    idx = 0
    for i, v in enumerate(values):
        total -= buf[idx]
        buf[idx] = float(v)
        total += buf[idx]
        idx = (idx + 1) % w
        out[i] = total / w
    return out


def compute_terms(ax, az, gy, k1, k2, d1, s_mm, coeff, alphav, window):
    h = np.sqrt((d1 * 1000.0) ** 2 - (s_mm / 2.0) ** 2)
    phi_crit = np.arctan((s_mm / 2.0) / h)
    wcrit = np.sqrt(coeff * (s_mm / 1000.0) * alphav / 4.0) * (360.0 / 6.28) * 1000.0
    az_safe = np.where(az == 0, 1e-9, az)
    phi_term = np.abs(np.arctan(ax / az_safe)) / phi_crit
    gy_avg = gy_ring_avg(gy, window)
    omega_term = (gy_avg / wcrit) ** 2
    si_theory = 1.0 - k1 * phi_term - k2 * omega_term
    return phi_term, omega_term, si_theory


def calc_metrics(si, pred):
    diff = pred - si
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(si, pred)[0, 1]) if len(si) > 1 else np.nan
    return mae, rmse, bias, corr


def main() -> None:
    a = parse_args()
    out = a.out or Path("output/results") / f"stability_debug_{a.csv.stem}.csv"

    df = pd.read_csv(a.csv)
    req = ["ax", "az", "gy", "si"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas requeridas: {miss}")

    num = df[req].apply(pd.to_numeric, errors="coerce")
    mask = num.notna().all(axis=1)
    if not mask.any():
        raise ValueError("No hay filas válidas con ax, az, gy y si numéricos.")

    ax = num.loc[mask, "ax"].to_numpy(float)
    az = num.loc[mask, "az"].to_numpy(float)
    gy = num.loc[mask, "gy"].to_numpy(float)
    si = num.loc[mask, "si"].to_numpy(float)

    phi_t, omg_t, si_th = compute_terms(ax, az, gy, a.k1, a.k2, a.d1, a.s_mm, a.coeff, a.alphav, a.window)
    si_diff = si_th - si

    df_out = df.copy()
    for c in ["phi_term", "omega_term", "si_theory", "si_diff"]:
        df_out[c] = np.nan
    df_out.loc[mask, "phi_term"] = phi_t
    df_out.loc[mask, "omega_term"] = omg_t
    df_out.loc[mask, "si_theory"] = si_th
    df_out.loc[mask, "si_diff"] = si_diff

    mae, rmse, bias, corr = calc_metrics(si, si_th)
    print(f"Filas usadas: {int(mask.sum())}/{len(df)}")
    print(f"MAE={mae:.6f} | RMSE={rmse:.6f} | Bias={bias:.6f} | Corr={corr:.4f}")

    top = df_out.loc[mask].assign(abs_diff=np.abs(si_diff)).nlargest(10, "abs_diff")
    cols = ["ax", "az", "gy", "si", "si_theory", "si_diff"]
    if "timestamp" in top.columns:
        cols = ["timestamp"] + cols
    print("\nTop 10 por |si_diff|:")
    print(top[cols].to_string(index=False))

    ax_scales, gy_scales = [1.0, 0.1, 0.01], [1.0, 1000.0, 0.001]
    ax_maes, gy_maes = [], []
    for s in ax_scales:
        pred = compute_terms(ax * s, az, gy, a.k1, a.k2, a.d1, a.s_mm, a.coeff, a.alphav, a.window)[2]
        ax_maes.append((s, float(np.mean(np.abs(pred - si)))))
    for s in gy_scales:
        pred = compute_terms(ax, az, gy * s, a.k1, a.k2, a.d1, a.s_mm, a.coeff, a.alphav, a.window)[2]
        gy_maes.append((s, float(np.mean(np.abs(pred - si)))))
    best_ax, best_ax_mae = min(ax_maes, key=lambda x: x[1])
    best_gy, best_gy_mae = min(gy_maes, key=lambda x: x[1])

    y = 1.0 - si
    x = np.column_stack([phi_t, omg_t])
    k1_fit, k2_fit = np.linalg.lstsq(x, y, rcond=None)[0]
    si_fit = 1.0 - k1_fit * phi_t - k2_fit * omg_t
    mae_fit = float(np.mean(np.abs(si_fit - si)))

    print("\nDiagnóstico de escala:")
    print(f"Mejor escala ax: {best_ax:g} (MAE={best_ax_mae:.6f})")
    print(f"Mejor escala gy: {best_gy:g} (MAE={best_gy_mae:.6f})")
    print("\nAjuste por mínimos cuadrados:")
    print(f"k1_fit={float(k1_fit):.6f}, k2_fit={float(k2_fit):.6f}, MAE_fit={mae_fit:.6f}")

    rel_k1 = abs((float(k1_fit) - a.k1) / a.k1) if a.k1 else np.inf
    rel_k2 = abs((float(k2_fit) - a.k2) / a.k2) if a.k2 else np.inf
    print("\nInterpretación rápida:")
    if best_ax != 1.0 or best_gy != 1.0:
        print("- Posible mismatch de unidades/escala en ax o gy.")
    if rel_k1 > 0.2 or rel_k2 > 0.2:
        print("- k1/k2 ajustados difieren >20%: revisar preferencias de firmware.")
    if np.isfinite(corr) and corr < 0.5:
        print("- Correlación baja: posible asincronía, latencia o ruido.")

    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\nCSV enriquecido guardado en: {out}")

    if a.plot:
        png = out.with_suffix(".png")
        try:
            import matplotlib.pyplot as plt

            xidx = np.arange(len(si))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ax1.plot(xidx, si, label="si", lw=1.1)
            ax1.plot(xidx, si_th, label="si_theory", lw=1.1)
            ax1.legend()
            ax1.set_ylabel("SI")
            ax2.plot(xidx, np.abs(si_diff), color="tab:red", lw=1.0)
            ax2.set_ylabel("|si_diff|")
            ax2.set_xlabel("Muestra")
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            plt.close(fig)
            print(f"PNG guardado en: {png}")
        except Exception as exc:
            print(f"Aviso: matplotlib no disponible o error al graficar ({exc}).")


if __name__ == "__main__":
    main()

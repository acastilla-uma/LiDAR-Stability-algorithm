"""Physics-layer rollover stability calculations.

The public final stability index is a margin: ``1`` means maximum stability
and lower values indicate increasing instability. Intermediate terms named
``*_risk`` increase with instability and are converted to the final margin by
``final_si_from_terms``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_VEHICLE_PARAMS = {
    "mass_kg": 18000.0,
    "track_width_m": 2.480,
    "cg_height_m": 1.850,
    "roll_inertia_kg_m2": 89300.0,
    "k_static": 1.0,
    "k_dynamic": 1.0,
    "omega_coeff": 1.0,
    "omega_correction_factor": 1.0,
    "omega_eps": 1e-9,
}


class StabilityEngine:
    """Compute static, dynamic, and final stability-index physics terms."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.params = DEFAULT_VEHICLE_PARAMS.copy()
        if config_path is not None:
            self.params.update(self._load_vehicle_params(Path(config_path)))

    def _load_vehicle_params(self, config_path: Path) -> dict[str, float]:
        if not config_path.exists():
            return {}

        with config_path.open("r", encoding="utf-8") as fh:
            payload: dict[str, Any] = yaml.safe_load(fh) or {}

        vehicle = payload.get("vehicle") or {}
        loaded: dict[str, float] = {}
        for key in DEFAULT_VEHICLE_PARAMS:
            value = vehicle.get(key)
            if value is not None:
                loaded[key] = float(value)
        return loaded

    def critical_angle(
        self,
        *,
        degrees: bool = True,
        track_width_m: float | None = None,
        cg_height_m: float | None = None,
    ) -> float:
        """Return critical rollover angle from track width and CG height."""
        s_m = self.params["track_width_m"] if track_width_m is None else float(track_width_m)
        h_m = self.params["cg_height_m"] if cg_height_m is None else float(cg_height_m)
        angle = self.critical_cross_slope_angle(s_m=s_m, h_m=h_m)
        return float(np.degrees(angle) if degrees else angle)

    def critical_cross_slope_angle(self, *, s_m: float | None = None, h_m: float | None = None) -> float:
        """Return ``phi_crit`` in radians: ``atan((s_m / 2) / h_m)``."""
        track_width = self.params["track_width_m"] if s_m is None else float(s_m)
        cg_height = self.params["cg_height_m"] if h_m is None else float(h_m)
        if track_width <= 0.0:
            raise ValueError("track width must be positive")
        if cg_height <= 0.0:
            raise ValueError("CG height must be positive")
        return float(np.arctan((track_width / 2.0) / cg_height))

    def static_lidar_risk(
        self,
        phi_lidar_rad: float | np.ndarray,
        *,
        phi_crit_rad: float | None = None,
        k1: float | None = None,
    ) -> float | np.ndarray:
        """Return static instability risk from cross-slope angle in radians."""
        phi_crit = self.critical_angle(degrees=False) if phi_crit_rad is None else float(phi_crit_rad)
        if phi_crit <= 0.0:
            raise ValueError("phi_crit_rad must be positive")
        weight = self.params["k_static"] if k1 is None else float(k1)
        values = weight * np.abs(np.asarray(phi_lidar_rad, dtype=float)) / phi_crit
        return float(values) if values.ndim == 0 else values

    def omega_critical(
        self,
        ay_m_s2: float | np.ndarray,
        *,
        coeff: float | None = None,
        s_m: float | None = None,
        correction_factor: float | None = None,
        eps: float | None = None,
    ) -> float | np.ndarray:
        """Return critical angular velocity in rad/s."""
        omega_coeff = self.params["omega_coeff"] if coeff is None else float(coeff)
        track_width = self.params["track_width_m"] if s_m is None else float(s_m)
        factor = self.params["omega_correction_factor"] if correction_factor is None else float(correction_factor)
        floor = self.params["omega_eps"] if eps is None else float(eps)
        if track_width <= 0.0:
            raise ValueError("track width must be positive")
        if omega_coeff < 0.0:
            raise ValueError("omega coefficient must be non-negative")
        if factor < 0.0:
            raise ValueError("correction factor must be non-negative")
        if floor <= 0.0:
            raise ValueError("eps must be positive")
        base = omega_coeff * track_width * np.abs(np.asarray(ay_m_s2, dtype=float)) * factor / 4.0
        values = np.sqrt(np.maximum(base, floor))
        return float(values) if values.ndim == 0 else values

    def dynamic_omega_risk(
        self,
        omega_rad_s: float | np.ndarray,
        omega_crit_rad_s: float | np.ndarray,
        *,
        k2: float | None = None,
    ) -> float | np.ndarray:
        """Return dynamic instability risk from predicted angular velocity."""
        omega_crit = np.asarray(omega_crit_rad_s, dtype=float)
        if np.any(omega_crit <= 0.0):
            raise ValueError("omega_crit_rad_s must be positive")
        weight = self.params["k_dynamic"] if k2 is None else float(k2)
        values = weight * (np.abs(np.asarray(omega_rad_s, dtype=float)) / omega_crit) ** 2
        return float(values) if values.ndim == 0 else values

    def final_si_from_terms(
        self,
        static_risk: float | np.ndarray,
        dynamic_risk: float | np.ndarray = 0.0,
        *,
        clip: bool = True,
    ) -> float | np.ndarray:
        """Return final SI margin where ``1`` is maximum stability."""
        risk_total = np.asarray(static_risk, dtype=float) + np.asarray(dynamic_risk, dtype=float)
        values = 1.0 - risk_total
        if clip:
            values = np.clip(values, 0.0, 1.0)
        return float(values) if values.ndim == 0 else values

    def compute_terms(
        self,
        *,
        phi_lidar_rad: float | np.ndarray,
        omega_rad_s: float | np.ndarray,
        ay_m_s2: float | np.ndarray,
        k1: float | None = None,
        k2: float | None = None,
        coeff: float | None = None,
        correction_factor: float | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Compute static risk, dynamic risk, total risk, and final SI margin."""
        phi_crit = self.critical_angle(degrees=False)
        omega_crit = self.omega_critical(
            ay_m_s2,
            coeff=coeff,
            correction_factor=correction_factor,
        )
        static_risk = self.static_lidar_risk(phi_lidar_rad, phi_crit_rad=phi_crit, k1=k1)
        dynamic_risk = self.dynamic_omega_risk(omega_rad_s, omega_crit, k2=k2)
        total_risk = np.asarray(static_risk, dtype=float) + np.asarray(dynamic_risk, dtype=float)
        si_pred = self.final_si_from_terms(total_risk, 0.0)
        return {
            "phi_crit_rad": phi_crit,
            "omega_crit_rad_s": omega_crit,
            "si_static_lidar_risk": static_risk,
            "si_dynamic_omega_risk": dynamic_risk,
            "si_risk_total": total_risk,
            "si_pred": si_pred,
        }

    def si_static(self, roll_rad: float) -> float:
        """Return legacy static risk/utilization for roll in radians."""
        return float(self.static_lidar_risk(roll_rad))

    def si_static_batch(self, roll_rad: np.ndarray) -> np.ndarray:
        """Vectorized legacy static risk/utilization for roll values in radians."""
        return np.asarray(self.static_lidar_risk(roll_rad), dtype=float)

    def si_static_from_deg(self, roll_deg: float) -> float:
        """Return static stability index for roll in degrees."""
        return self.si_static(float(np.radians(roll_deg)))

    def get_vehicle_params(self) -> dict[str, float]:
        """Return the active vehicle parameters."""
        return self.params.copy()

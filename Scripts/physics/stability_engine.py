"""
Stability Engine - Physics Module

Implements deterministic physics model for vehicle rollover stability.
Calculates critical angle and static stability index (SI_static).
"""

import numpy as np
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StabilityEngine:
    """
    Deterministic physics model for rollover stability.
    
    Theory:
    - Critical rollover angle: φc = arctan(S / (2*Hg))
    - Stability Index: SI = 1 - k1*(|φ|/φc) - k2*(|ω|/ωc)^2
    - SI = 1.0 means fully stable
    - SI can be < 1 as instability increases
    
    Firmware-aligned (cabina_v2_4_raspberry.ino):
    - h = sqrt((d1*1000)^2 - (s/2)^2)
    - φcrit = atan((s/2)/h)
    - φ = |atan(ax/az)|
    - wcrit = sqrt(coeff*(s/1000)*alphav/4) * (360/6.28) * 1000  [mdeg/s]
    - SI = 1 - k1*(φ/φcrit) - k2*(gy_avg/wcrit)^2
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize engine with vehicle configuration.
        
        Args:
            config_path: Path to vehicle.yaml config file
        """
        self.config = self._load_config(config_path)
        self._compute_critical_angle()
        self._load_stability_model_params()
        self._load_empirical_params()
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load vehicle configuration from YAML."""
        if config_path is None:
            # Try default location
            config_path = Path(__file__).parent.parent / 'config' / 'vehicle.yaml'
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded vehicle config: {config_path}")
        return config
    
    def _compute_critical_angle(self):
        """Compute critical rollover angle from vehicle geometry."""
        vehicle = self.config['vehicle']
        S = vehicle['track_width_m']  # Track width
        Hg = vehicle['cg_height_m']   # CG height
        
        # φc = arctan(S / (2*Hg))
        self.phi_c_rad = np.arctan(S / (2 * Hg))
        self.phi_c_deg = np.degrees(self.phi_c_rad)
        
        logger.info(f"Critical rollover angle φc = {self.phi_c_deg:.2f}° "
                   f"({self.phi_c_rad:.4f} rad)")

    def _load_stability_model_params(self):
        """Load SI model coefficients from config with safe defaults."""
        model = self.config.get('stability_model', {})

        self.k1 = float(model.get('k1', 1.15))
        self.k2 = float(model.get('k2', 2.05))
        self.d1_m = float(model.get('d1_m', 4.2))
        self.s_mm = float(model.get('s_mm', 1100.0))
        self.k4_mm = float(model.get('k4_mm', 1100.0))
        self.coeff = float(model.get('coeff', 7.14))
        self.alphav = float(model.get('alphav', 64.0))
        self.gy_avg_window = int(model.get('gy_avg_window', 10))

        if self.d1_m <= 0:
            raise ValueError("d1_m must be > 0")
        if self.s_mm <= 0:
            raise ValueError("s_mm must be > 0")
        if self.coeff <= 0 or self.alphav <= 0:
            raise ValueError("coeff and alphav must be > 0")
        if self.gy_avg_window <= 0:
            raise ValueError("gy_avg_window must be > 0")

        half_s_mm = self.s_mm / 2.0
        d1_mm = self.d1_m * 1000.0
        if d1_mm <= half_s_mm:
            raise ValueError("d1_m is too small for s_mm (invalid geometry)")

        self.h_mm = float(np.sqrt((d1_mm ** 2) - (half_s_mm ** 2)))
        self.phi_crit_ino_rad = float(np.arctan(half_s_mm / self.h_mm))
        self.phi_crit_ino_deg = float(np.degrees(self.phi_crit_ino_rad))

        half_k4_mm = self.k4_mm / 2.0
        self.phi_crit_front_rad = float(np.arctan(half_k4_mm / self.h_mm))
        self.phi_crit_front_deg = float(np.degrees(self.phi_crit_front_rad))

        self.wcrit_mdeg_s = float(
            np.sqrt(self.coeff * (self.s_mm / 1000.0) * self.alphav / 4.0) * (360.0 / 6.28) * 1000.0
        )

        self.omega_crit_rad_s = float(np.radians(self.wcrit_mdeg_s / 1000.0))

        if self.omega_crit_rad_s <= 0:
            raise ValueError("omega_crit_rad_s must be > 0")

        logger.info(
            "Stability model params loaded: "
            f"k1={self.k1:.4f}, k2={self.k2:.4f}, "
            f"phi_crit_ino={self.phi_crit_ino_deg:.2f} deg, "
            f"wcrit={self.wcrit_mdeg_s:.2f} mdeg/s"
        )

    def _load_empirical_params(self):
        """Load empirical static-fusion parameters used in sprint 5."""
        model = self.config.get('stability_model', {})
        self.static_fusion_alpha = float(model.get('static_fusion_alpha', 0.35))
        self.tri_gain = float(model.get('tri_gain', 0.15))
        self.tri_clip = float(model.get('tri_clip', 3.0))

        self.static_fusion_alpha = float(np.clip(self.static_fusion_alpha, 0.0, 1.0))
        self.tri_gain = max(0.0, self.tri_gain)
        self.tri_clip = max(0.0, self.tri_clip)
    
    def critical_angle(self, degrees: bool = True) -> float:
        """
        Get critical rollover angle.
        
        Args:
            degrees: Return in degrees (True) or radians (False)
            
        Returns:
            Critical angle
        """
        return self.phi_c_deg if degrees else self.phi_c_rad
    
    def si_static(self, roll_rad: float, omega_rad_s: float = 0.0) -> float:
        """
        Calculate stability index from roll and angular rate.
        
        Args:
            roll_rad: Roll angle in radians
            omega_rad_s: Angular rate in rad/s
            
        Returns:
            SI value (1 = fully stable, lower values imply lower stability)
        """
        if self.phi_c_rad == 0 or self.omega_crit_rad_s == 0:
            return 0.0

        phi_term = abs(float(roll_rad)) / self.phi_c_rad
        omega_term = (abs(float(omega_rad_s)) / self.omega_crit_rad_s) ** 2

        si = 1.0 - self.k1 * phi_term - self.k2 * omega_term
        return float(si)

    def si_ino(self, ax: float, az: float, gy_mdeg_s: float = 0.0) -> float:
        """
        Calculate SI exactly as firmware does for lateral rollover term.

        Args:
            ax: lateral acceleration component (same units as az)
            az: vertical acceleration component
            gy_mdeg_s: angular rate around y-axis in mdeg/s

        Returns:
            SI value
        """
        az_val = float(az)
        if abs(az_val) < 1e-9:
            az_val = 1e-9

        phi = abs(float(np.arctan(float(ax) / az_val)))
        phi_term = phi / self.phi_crit_ino_rad
        omega_term = (float(gy_mdeg_s) / self.wcrit_mdeg_s) ** 2
        si = 1.0 - self.k1 * phi_term - self.k2 * omega_term
        return float(si)

    def si_ino_batch(self, ax_array: np.ndarray, az_array: np.ndarray, gy_mdeg_s_array: np.ndarray = None,
                     smooth_window: int = None) -> np.ndarray:
        """
        Vectorized firmware-aligned SI computation.

        Args:
            ax_array: Lateral acceleration array
            az_array: Vertical acceleration array
            gy_mdeg_s_array: Angular-rate array in mdeg/s (optional)
            smooth_window: Moving-average window for gy (defaults to gy_avg_window)
        """
        ax_array = np.asarray(ax_array, dtype=float)
        az_array = np.asarray(az_array, dtype=float)
        if gy_mdeg_s_array is None:
            gy_mdeg_s_array = np.zeros_like(ax_array, dtype=float)
        else:
            gy_mdeg_s_array = np.asarray(gy_mdeg_s_array, dtype=float)

        if smooth_window is None:
            smooth_window = self.gy_avg_window
        smooth_window = max(1, int(smooth_window))

        if smooth_window > 1 and len(gy_mdeg_s_array) > 0:
            buffer = np.zeros(smooth_window, dtype=float)
            gy_avg = np.zeros_like(gy_mdeg_s_array, dtype=float)
            index = 0
            for i, value in enumerate(gy_mdeg_s_array):
                buffer[index] = value
                index = (index + 1) % smooth_window
                gy_avg[i] = np.mean(buffer)
        else:
            gy_avg = gy_mdeg_s_array

        az_safe = np.where(np.abs(az_array) < 1e-9, 1e-9, az_array)
        phi = np.abs(np.arctan(ax_array / az_safe))
        phi_term = phi / self.phi_crit_ino_rad
        omega_term = (gy_avg / self.wcrit_mdeg_s) ** 2

        return 1.0 - self.k1 * phi_term - self.k2 * omega_term
    
    def si_static_batch(self, roll_array: np.ndarray, omega_array: np.ndarray = None) -> np.ndarray:
        """
        Vectorized SI computation.
        
        Args:
            roll_array: Array of roll angles in radians
            omega_array: Array of angular rates in rad/s (optional, defaults to zeros)
            
        Returns:
            Array of SI values
        """
        roll_array = np.asarray(roll_array, dtype=float)
        if omega_array is None:
            omega_array = np.zeros_like(roll_array, dtype=float)
        else:
            omega_array = np.asarray(omega_array, dtype=float)

        phi_term = np.abs(roll_array) / self.phi_c_rad
        omega_term = (np.abs(omega_array) / self.omega_crit_rad_s) ** 2

        si = 1.0 - self.k1 * phi_term - self.k2 * omega_term
        return si

    def effective_roll_from_lidar(self,
                                  roll_rad: float,
                                  phi_lidar_rad: float,
                                  tri: float = 0.0,
                                  static_fusion_alpha: float = None,
                                  tri_gain: float = None,
                                  tri_clip: float = None) -> float:
        """Blend IMU roll and LiDAR slope into an empirical effective roll."""
        alpha = self.static_fusion_alpha if static_fusion_alpha is None else float(static_fusion_alpha)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        tri_gain_val = self.tri_gain if tri_gain is None else max(0.0, float(tri_gain))
        tri_clip_val = self.tri_clip if tri_clip is None else max(0.0, float(tri_clip))

        roll_val = float(roll_rad) if np.isfinite(roll_rad) else 0.0
        phi_val = float(phi_lidar_rad) if np.isfinite(phi_lidar_rad) else roll_val
        tri_val = float(tri) if np.isfinite(tri) else 0.0

        tri_factor = 1.0 + tri_gain_val * min(max(tri_val, 0.0), tri_clip_val)
        fused_abs = (1.0 - alpha) * abs(roll_val) + alpha * abs(phi_val) * tri_factor
        sign = np.sign(roll_val) if abs(roll_val) > 1e-9 else np.sign(phi_val)
        return float(sign * fused_abs)

    def effective_roll_from_lidar_batch(self,
                                        roll_rad_array: np.ndarray,
                                        phi_lidar_rad_array: np.ndarray,
                                        tri_array: np.ndarray = None,
                                        static_fusion_alpha: float = None,
                                        tri_gain: float = None,
                                        tri_clip: float = None) -> np.ndarray:
        """Vectorized effective roll blending."""
        roll_arr = np.asarray(roll_rad_array, dtype=float)
        phi_arr = np.asarray(phi_lidar_rad_array, dtype=float)

        if tri_array is None:
            tri_arr = np.zeros_like(roll_arr, dtype=float)
        else:
            tri_arr = np.asarray(tri_array, dtype=float)

        alpha = self.static_fusion_alpha if static_fusion_alpha is None else float(static_fusion_alpha)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        tri_gain_val = self.tri_gain if tri_gain is None else max(0.0, float(tri_gain))
        tri_clip_val = self.tri_clip if tri_clip is None else max(0.0, float(tri_clip))

        tri_clean = np.where(np.isfinite(tri_arr), tri_arr, 0.0)
        tri_clean = np.clip(tri_clean, 0.0, tri_clip_val)
        tri_factor = 1.0 + tri_gain_val * tri_clean

        roll_clean = np.where(np.isfinite(roll_arr), roll_arr, 0.0)
        phi_clean = np.where(np.isfinite(phi_arr), phi_arr, roll_clean)

        fused_abs = (1.0 - alpha) * np.abs(roll_clean) + alpha * np.abs(phi_clean) * tri_factor
        sign = np.where(np.abs(roll_clean) > 1e-9, np.sign(roll_clean), np.sign(phi_clean))
        return sign * fused_abs

    def si_static_empirical(self,
                            roll_rad: float,
                            phi_lidar_rad: float,
                            tri: float = 0.0,
                            static_fusion_alpha: float = None,
                            tri_gain: float = None,
                            tri_clip: float = None) -> float:
        """Compute static SI from LiDAR-enhanced effective roll."""
        eff_roll = self.effective_roll_from_lidar(
            roll_rad=roll_rad,
            phi_lidar_rad=phi_lidar_rad,
            tri=tri,
            static_fusion_alpha=static_fusion_alpha,
            tri_gain=tri_gain,
            tri_clip=tri_clip,
        )
        return self.si_static(eff_roll, omega_rad_s=0.0)

    def si_static_empirical_batch(self,
                                  roll_rad_array: np.ndarray,
                                  phi_lidar_rad_array: np.ndarray,
                                  tri_array: np.ndarray = None,
                                  static_fusion_alpha: float = None,
                                  tri_gain: float = None,
                                  tri_clip: float = None) -> np.ndarray:
        """Vectorized static SI from LiDAR-enhanced effective roll."""
        eff_roll = self.effective_roll_from_lidar_batch(
            roll_rad_array=roll_rad_array,
            phi_lidar_rad_array=phi_lidar_rad_array,
            tri_array=tri_array,
            static_fusion_alpha=static_fusion_alpha,
            tri_gain=tri_gain,
            tri_clip=tri_clip,
        )
        return self.si_static_batch(eff_roll)

    def dynamic_penalty_from_w(self, omega_rad_s: float) -> float:
        """Compute dynamic SI penalty term from angular velocity."""
        if self.omega_crit_rad_s <= 0:
            return 0.0
        omega_term = (abs(float(omega_rad_s)) / self.omega_crit_rad_s) ** 2
        return float(self.k2 * omega_term)

    def dynamic_penalty_from_w_batch(self, omega_rad_s_array: np.ndarray) -> np.ndarray:
        """Vectorized dynamic SI penalty from angular velocity."""
        omega_arr = np.asarray(omega_rad_s_array, dtype=float)
        if self.omega_crit_rad_s <= 0:
            return np.zeros_like(omega_arr, dtype=float)
        omega_term = (np.abs(omega_arr) / self.omega_crit_rad_s) ** 2
        return self.k2 * omega_term

    def si_combined_batch(self,
                          roll_rad_array: np.ndarray,
                          omega_rad_s_array: np.ndarray,
                          phi_lidar_rad_array: np.ndarray = None,
                          tri_array: np.ndarray = None,
                          static_fusion_alpha: float = None,
                          tri_gain: float = None,
                          tri_clip: float = None) -> np.ndarray:
        """Compute SI using static (IMU/LiDAR fused) plus dynamic omega terms."""
        roll_arr = np.asarray(roll_rad_array, dtype=float)
        omega_arr = np.asarray(omega_rad_s_array, dtype=float)

        if phi_lidar_rad_array is None:
            static_roll = roll_arr
        else:
            static_roll = self.effective_roll_from_lidar_batch(
                roll_rad_array=roll_arr,
                phi_lidar_rad_array=np.asarray(phi_lidar_rad_array, dtype=float),
                tri_array=tri_array,
                static_fusion_alpha=static_fusion_alpha,
                tri_gain=tri_gain,
                tri_clip=tri_clip,
            )

        static_penalty = self.k1 * (np.abs(static_roll) / self.phi_c_rad)
        dynamic_penalty = self.dynamic_penalty_from_w_batch(omega_arr)
        return 1.0 - static_penalty - dynamic_penalty
    
    def si_static_from_deg(self, roll_deg: float, omega_deg_s: float = 0.0) -> float:
        """
        Calculate SI from roll angle in degrees.
        
        Args:
            roll_deg: Roll angle in degrees
            omega_deg_s: Angular rate in deg/s
            
        Returns:
            SI value
        """
        roll_rad = np.radians(roll_deg)
        omega_rad_s = np.radians(omega_deg_s)
        return self.si_static(roll_rad, omega_rad_s)
    
    def si_static_batch_from_deg(self, roll_deg_array: np.ndarray, omega_deg_s_array: np.ndarray = None) -> np.ndarray:
        """
        Vectorized SI computation from degrees.
        
        Args:
            roll_deg_array: Array of roll angles in degrees
            omega_deg_s_array: Array of angular rates in deg/s (optional)
            
        Returns:
            Array of SI values
        """
        roll_rad_array = np.radians(roll_deg_array)
        omega_rad_s_array = None if omega_deg_s_array is None else np.radians(omega_deg_s_array)
        return self.si_static_batch(roll_rad_array, omega_rad_s_array)
    
    def get_vehicle_params(self) -> dict:
        """Return vehicle parameters for inspection."""
        vehicle = self.config['vehicle']
        return {
            'mass_kg': vehicle['mass_kg'],
            'track_width_m': vehicle['track_width_m'],
            'cg_height_m': vehicle['cg_height_m'],
            'roll_inertia_kg_m2': vehicle['roll_inertia_kg_m2'],
            'suspension_type': vehicle['suspension_type'],
            'phi_c_deg': self.phi_c_deg,
            'phi_c_rad': self.phi_c_rad,
            'd1_m': self.d1_m,
            's_mm': self.s_mm,
            'k4_mm': self.k4_mm,
            'h_mm': self.h_mm,
            'phi_crit_ino_deg': self.phi_crit_ino_deg,
            'phi_crit_front_deg': self.phi_crit_front_deg,
            'k1': self.k1,
            'k2': self.k2,
            'coeff': self.coeff,
            'alphav': self.alphav,
            'wcrit_mdeg_s': self.wcrit_mdeg_s,
            'omega_crit_rad_s': self.omega_crit_rad_s,
            'static_fusion_alpha': self.static_fusion_alpha,
            'tri_gain': self.tri_gain,
            'tri_clip': self.tri_clip,
        }
    
    def get_stability_thresholds(self) -> dict:
        """Return stability threshold values."""
        return self.config['stability']

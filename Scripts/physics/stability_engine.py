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
    - Static Stability Index: SI = tan(φ_roll) / tan(φ_c)
    - When SI >= 1.0, vehicle may rollover
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize engine with vehicle configuration.
        
        Args:
            config_path: Path to vehicle.yaml config file
        """
        self.config = self._load_config(config_path)
        self._compute_critical_angle()
    
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
    
    def critical_angle(self, degrees: bool = True) -> float:
        """
        Get critical rollover angle.
        
        Args:
            degrees: Return in degrees (True) or radians (False)
            
        Returns:
            Critical angle
        """
        return self.phi_c_deg if degrees else self.phi_c_rad
    
    def si_static(self, roll_rad: float) -> float:
        """
        Calculate static stability index from roll angle.
        
        Args:
            roll_rad: Roll angle in radians
            
        Returns:
            SI_static: Stability index (0 = stable, 1 = critical, >1 = unstable)
        """
        # Avoid division by zero
        if self.phi_c_rad == 0:
            return 0.0
        
        # SI = tan(φ_roll) / tan(φ_c)
        if roll_rad == 0:
            return 0.0
        
        tan_roll = np.tan(roll_rad)
        tan_phi_c = np.tan(self.phi_c_rad)
        
        si = tan_roll / tan_phi_c
        return float(si)
    
    def si_static_batch(self, roll_array: np.ndarray) -> np.ndarray:
        """
        Vectorized static SI computation.
        
        Args:
            roll_array: Array of roll angles in radians
            
        Returns:
            Array of SI values
        """
        tan_roll = np.tan(roll_array)
        tan_phi_c = np.tan(self.phi_c_rad)
        
        si = tan_roll / tan_phi_c
        return si
    
    def si_static_from_deg(self, roll_deg: float) -> float:
        """
        Calculate SI from roll angle in degrees.
        
        Args:
            roll_deg: Roll angle in degrees
            
        Returns:
            SI_static value
        """
        roll_rad = np.radians(roll_deg)
        return self.si_static(roll_rad)
    
    def si_static_batch_from_deg(self, roll_deg_array: np.ndarray) -> np.ndarray:
        """
        Vectorized SI computation from degrees.
        
        Args:
            roll_deg_array: Array of roll angles in degrees
            
        Returns:
            Array of SI values
        """
        roll_rad_array = np.radians(roll_deg_array)
        return self.si_static_batch(roll_rad_array)
    
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
            'phi_c_rad': self.phi_c_rad
        }
    
    def get_stability_thresholds(self) -> dict:
        """Return stability threshold values."""
        return self.config['stability']

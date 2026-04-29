"""
Device Configuration Registry

Manages device-specific configurations (K1, K2, D1, S, Coeff, AlphaV, etc.)
for different DOBACK units (23, 24, 27, 28).

Ejemplo de uso:
    registry = DeviceRegistry()
    config = registry.get_config("24")  # DOBACK024
    k1 = config['stability_model']['k1']
    
    device_id = registry.get_device_from_filename("DOBACK024_20251001_seg3.csv")
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

logger = logging.getLogger(__name__)


class DeviceRegistry:
    """Registry para configuraciones específicas por dispositivo."""
    
    def __init__(self, devices_dir: Optional[Path] = None):
        """
        Inicializar registry cargando YAMLs de dispositivos.
        
        Args:
            devices_dir: Directorio con archivos YAML por device.
                        Default: src/lidar_stability/config/devices/
        """
        if devices_dir is None:
            # Detectar automáticamente directorio
            current_file = Path(__file__)
            devices_dir = current_file.parent / "devices"
        
        self.devices_dir = Path(devices_dir)
        self._configs = {}
        self._device_map = {}  # Mapeo device_id -> nombre config
        
        # Cargar todos los YAMLs disponibles
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Cargar todos los archivos YAML en devices/."""
        if not self.devices_dir.exists():
            logger.warning(f"Directorio devices no encontrado: {self.devices_dir}")
            return
        
        for yaml_file in sorted(self.devices_dir.glob("doback-*.yaml")):
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config is None:
                    logger.warning(f"YAML vacío: {yaml_file}")
                    continue
                
                device_id = config.get('device_id')
                if not device_id:
                    logger.warning(f"YAML sin device_id: {yaml_file}")
                    continue
                
                self._configs[device_id] = config
                self._device_map[f"DOBACK{device_id}"] = device_id
                logger.info(f"Cargado: DOBACK{device_id} desde {yaml_file.name}")
                
            except Exception as e:
                logger.error(f"Error cargando {yaml_file}: {e}")
    
    def get_config(self, device_id: str) -> Dict:
        """
        Obtener configuración para un dispositivo.
        
        Args:
            device_id: ID del device ("23", "24", "27", "28")
        
        Returns:
            Dict con configuración del device
            
        Raises:
            ValueError: Si device_id no tiene configuración
        """
        # Normalizar device_id (quitar "DOBACK" si existe)
        device_id = str(device_id).replace("DOBACK", "").strip()
        
        if device_id not in self._configs:
            raise ValueError(
                f"Device {device_id} no tiene configuración. "
                f"Disponibles: {list(self._configs.keys())}"
            )
        
        return self._configs[device_id]
    
    def get_device_from_filename(self, filename: str) -> Optional[str]:
        """
        Extraer device_id de nombre de archivo.
        
        Ejemplos:
            "DOBACK024_20251001_seg3.csv" -> "24"
            "GPS_DOBACK027_20250929.txt" -> "27"
            "ESTABILIDAD_DOBACK023_20251012.txt" -> "23"
        
        Args:
            filename: Nombre de archivo
        
        Returns:
            Device ID sin prefix ("24") o None si no se encuentra
        """
        filename_upper = str(filename).upper()
        
        # Buscar patrón DOBACK + dígitos
        for part in filename_upper.split("_"):
            if part.startswith("DOBACK") and len(part) >= 8:
                # Extraer números después de DOBACK
                digits = part[6:9]  # DOBACK + 3 dígitos
                if digits.isdigit():
                    return digits[-2:]  # Retornar últimos 2 dígitos
        
        return None
    
    def list_devices(self) -> list:
        """Listar todos los device_ids disponibles."""
        return sorted(self._configs.keys())
    
    def validate_constants(self, device_id: str) -> bool:
        """
        Validar que la configuración tenga todos los campos requeridos.
        
        Args:
            device_id: ID del device
        
        Returns:
            True si configuración es válida
        """
        try:
            config = self.get_config(device_id)
            required_fields = ['stability_model', 'terrain', 'hardware']
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Campo faltante en DOBACK{device_id}: {field}")
                    return False
            
            required_stability = ['k1', 'k2', 'd1_m', 's_mm', 'coeff', 'alphav']
            stability = config.get('stability_model', {})
            for field in required_stability:
                if field not in stability:
                    logger.warning(f"Campo de stability faltante en DOBACK{device_id}: {field}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validando DOBACK{device_id}: {e}")
            return False


# Instancia global (lazy loading)
_registry = None


def get_registry() -> DeviceRegistry:
    """Obtener instancia global del registry."""
    global _registry
    if _registry is None:
        _registry = DeviceRegistry()
    return _registry

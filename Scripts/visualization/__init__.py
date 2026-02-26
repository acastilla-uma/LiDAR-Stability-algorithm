"""
Módulo de visualización para el proyecto LiDAR Stability Algorithm.

Este módulo contiene herramientas para visualizar:
- Nubes de puntos LAZ (CNIG PNOA)
- Rasters TIF (Digital Terrain Models)
- Trayectorias GPS con datos de estabilidad
- Dashboards integrados

Autor: Alex Castilla
Fecha: 2025-02-24
"""

__version__ = '1.0.0'
__author__ = 'Alex Castilla'

# Importaciones principales
from . import visualize_laz
from . import visualize_tif
from . import visualize_gps_stability
from . import visualize_all

__all__ = [
    'visualize_laz',
    'visualize_tif',
    'visualize_gps_stability',
    'visualize_all'
]

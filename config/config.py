"""
Archivo de configuración para el procesamiento de datos GPS
"""

import os

# Estructura de directorios del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios de datos
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "gps")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed", "gps")

# Directorios de salida
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps", "gps")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Directorio de datos legacy (mantener compatibilidad)
LEGACY_DATA_DIR = os.path.join(PROJECT_ROOT, "Doback-Data")

# Parámetros de validación GPS
GPS_VALIDATION = {
    # Coordenadas esperadas para Alcobendas, Madrid
    "expected_location": {
        "latitude": 40.548,
        "longitude": -3.641,
        "name": "Alcobendas, Madrid"
    },
    
    # Rangos válidos absolutos
    "valid_ranges": {
        "latitude": {"min": -90, "max": 90},
        "longitude": {"min": -180, "max": 180},
        "altitude": {"min": -500, "max": 5000},  # metros
        "speed": {"min": 0, "max": 200}  # km/h
    },
    
    # Tolerancia para detección de outliers (km desde ubicación esperada)
    "max_distance_from_expected": 50,  # 50 km de Alcobendas
    
    # Desviación estándar máxima para detección de outliers
    "outlier_std_threshold": 3.0,
    
    # Mínimo de satélites para considerar válida la lectura
    "min_satellites": 4,
    
    # HDOP máximo (precisión horizontal)
    "max_hdop": 5.0
}

# Configuración de mapas
MAP_CONFIG = {
    "default_zoom": 15,
    "tile_layers": [
        "OpenStreetMap",
        "Cartodb Positron",
        "Cartodb dark_matter"
    ],
    "route_color": "#2E86DE",
    "route_weight": 3,
    "route_opacity": 0.8,
    
    # Colores para marcadores
    "colors": {
        "start": "green",
        "end": "red",
        "waypoint": "blue",
        "outlier": "orange"
    }
}

# Configuración de procesamiento
PROCESSING_CONFIG = {
    # Frecuencia de marcadores en el mapa (1 cada N puntos)
    "marker_frequency": 50,
    
    # Guardar datos procesados
    "save_processed_data": True,
    
    # Generar reporte estadístico
    "generate_report": True,
    
    # Formato de salida
    "output_formats": {
        "map": "html",
        "data": "csv",
        "report": "txt"
    }
}

# Logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

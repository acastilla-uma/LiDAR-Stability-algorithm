"""
Script de ejemplo para probar las herramientas de visualización.

Este script ejecuta visualizaciones de ejemplo utilizando los datos
disponibles en el proyecto.

Autor: Alex Castilla
Fecha: 2025-02-24
"""

import os
import sys
import glob

# Rutas base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAZ_DIR = os.path.join(BASE_DIR, "LiDAR-Maps", "cnig")
GPS_DIR = os.path.join(BASE_DIR, "Doback-Data", "GPS")
STABILITY_DIR = os.path.join(BASE_DIR, "Doback-Data", "Stability")


def check_data_availability():
    """Verifica qué datos están disponibles."""
    print("=" * 70)
    print("VERIFICANDO DISPONIBILIDAD DE DATOS")
    print("=" * 70)
    
    # Verificar LAZ files
    laz_files = glob.glob(os.path.join(LAZ_DIR, "*.laz"))
    print(f"\n✓ Archivos LAZ encontrados: {len(laz_files)}")
    if laz_files:
        print(f"  Ejemplo: {os.path.basename(laz_files[0])}")
    
    # Verificar GPS files
    gps_files = glob.glob(os.path.join(GPS_DIR, "*.txt"))
    print(f"\n✓ Archivos GPS encontrados: {len(gps_files)}")
    if gps_files:
        for gps in gps_files[:2]:
            print(f"  - {os.path.basename(gps)}")
    
    # Verificar Stability files
    stab_files = glob.glob(os.path.join(STABILITY_DIR, "*.txt"))
    print(f"\n✓ Archivos de Estabilidad encontrados: {len(stab_files)}")
    if stab_files:
        for stab in stab_files[:2]:
            print(f"  - {os.path.basename(stab)}")
    
    print("\n" + "=" * 70)
    
    return laz_files, gps_files, stab_files


def example_laz_visualization(laz_file):
    """Ejemplo de visualización LAZ."""
    print("\n" + "=" * 70)
    print("EJEMPLO 1: VISUALIZACIÓN DE NUBE DE PUNTOS LAZ")
    print("=" * 70)
    
    from visualize_laz import visualize_laz_2d, visualize_laz_3d
    
    print(f"\nArchivo: {os.path.basename(laz_file)}")
    print("\n1a) Vista 2D cenital (50% de puntos)...")
    
    try:
        fig, ax = visualize_laz_2d(laz_file, color_by='elevation', sample_rate=0.5)
        print("   ✓ Vista 2D generada correctamente")
        print("   → Cierre la ventana para continuar...")
    except Exception as e:
        print(f"   ✗ Error: {e}")


def example_gps_stability_visualization(gps_file, stab_file):
    """Ejemplo de visualización GPS + Estabilidad."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: VISUALIZACIÓN GPS + ESTABILIDAD")
    print("=" * 70)
    
    from visualize_gps_stability import load_gps_data, load_stability_data, merge_gps_stability, visualize_trajectory_stability
    
    print(f"\nGPS: {os.path.basename(gps_file)}")
    print(f"Estabilidad: {os.path.basename(stab_file)}")
    
    try:
        # Cargar datos
        gps_df = load_gps_data(gps_file)
        stab_df = load_stability_data(stab_file)
        
        # Fusionar
        merged_df = merge_gps_stability(gps_df, stab_df, time_tolerance_us=1000000)
        
        if len(merged_df) > 0:
            print(f"\n✓ Datos fusionados: {len(merged_df)} registros")
            
            print("\n2a) Generando visualización de trayectoria con estabilidad...")
            fig, axes = visualize_trajectory_stability(merged_df)
            print("   ✓ Visualización generada correctamente")
            print("   → Cierre la ventana para continuar...")
        else:
            print("\n✗ No se pudieron fusionar los datos (timestamps incompatibles)")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")


def example_dashboard(laz_dir, gps_file, stab_file):
    """Ejemplo de dashboard integrado."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: DASHBOARD INTEGRADO")
    print("=" * 70)
    
    from visualize_all import load_gps_stability_data, dashboard_lidar_trajectory
    
    print(f"\nLiDAR: {laz_dir}")
    print(f"GPS: {os.path.basename(gps_file)}")
    print(f"Estabilidad: {os.path.basename(stab_file)}")
    
    try:
        # Cargar datos GPS + Estabilidad
        merged_df = load_gps_stability_data(gps_file, stab_file, time_tolerance_us=1000000)
        
        if len(merged_df) > 0:
            print(f"\n✓ Datos fusionados: {len(merged_df)} registros")
            
            print("\n3a) Generando dashboard integrado (puede tardar unos segundos)...")
            print("    Muestreando 5% de puntos LiDAR para rendimiento...")
            
            fig = dashboard_lidar_trajectory(laz_dir, merged_df, sample_rate=0.05)
            print("   ✓ Dashboard generado correctamente")
            print("   → Cierre la ventana para finalizar...")
        else:
            print("\n✗ No se pudieron fusionar los datos")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")


def print_menu():
    """Imprime el menú de opciones."""
    print("\n" + "=" * 70)
    print("MENÚ DE EJEMPLOS DE VISUALIZACIÓN")
    print("=" * 70)
    print("\n1. Visualizar nube de puntos LAZ (2D)")
    print("2. Visualizar GPS + Estabilidad")
    print("3. Dashboard integrado LiDAR + GPS + Estabilidad")
    print("4. Ejecutar todos los ejemplos")
    print("0. Salir")
    print("\n" + "=" * 70)


def main():
    """Función principal."""
    
    print("\n" + "=" * 70)
    print("EJEMPLOS DE VISUALIZACIÓN - LiDAR Stability Algorithm")
    print("=" * 70)
    print("\nEste script ejecuta ejemplos de las herramientas de visualización.")
    print("Las visualizaciones se mostrarán en ventanas interactivas.")
    print("Cierre cada ventana para continuar con el siguiente ejemplo.")
    
    # Verificar datos disponibles
    laz_files, gps_files, stab_files = check_data_availability()
    
    if not laz_files:
        print("\n⚠ No se encontraron archivos LAZ en LiDAR-Maps/cnig/")
        print("   Algunos ejemplos no estarán disponibles.")
    
    if not gps_files or not stab_files:
        print("\n⚠ No se encontraron archivos GPS o Estabilidad en Doback-Data/")
        print("   Algunos ejemplos no estarán disponibles.")
    
    # Seleccionar archivos de ejemplo
    laz_file = laz_files[0] if laz_files else None
    gps_file = gps_files[0] if gps_files else None
    stab_file = stab_files[0] if stab_files else None
    
    while True:
        print_menu()
        choice = input("\nSeleccione una opción [0-4]: ").strip()
        
        if choice == '0':
            print("\n✓ Saliendo...")
            break
        
        elif choice == '1':
            if laz_file:
                example_laz_visualization(laz_file)
            else:
                print("\n✗ No hay archivos LAZ disponibles")
        
        elif choice == '2':
            if gps_file and stab_file:
                example_gps_stability_visualization(gps_file, stab_file)
            else:
                print("\n✗ No hay archivos GPS o Estabilidad disponibles")
        
        elif choice == '3':
            if LAZ_DIR and gps_file and stab_file:
                example_dashboard(LAZ_DIR, gps_file, stab_file)
            else:
                print("\n✗ Faltan datos necesarios para el dashboard")
        
        elif choice == '4':
            print("\n▶ Ejecutando todos los ejemplos...")
            
            if laz_file:
                example_laz_visualization(laz_file)
            else:
                print("\n⊘ Omitiendo ejemplo LAZ (no disponible)")
            
            if gps_file and stab_file:
                example_gps_stability_visualization(gps_file, stab_file)
            else:
                print("\n⊘ Omitiendo ejemplo GPS+Estabilidad (no disponible)")
            
            if LAZ_DIR and gps_file and stab_file:
                example_dashboard(LAZ_DIR, gps_file, stab_file)
            else:
                print("\n⊘ Omitiendo dashboard integrado (no disponible)")
            
            print("\n✓ Todos los ejemplos completados")
        
        else:
            print("\n✗ Opción no válida")
    
    print("\n" + "=" * 70)
    print("EJEMPLOS FINALIZADOS")
    print("=" * 70)
    print("\nPara más opciones, consulte README.md en scripts/visualization/")
    print("O ejecute los scripts directamente desde la línea de comandos.\n")


if __name__ == '__main__':
    main()

"""
Visualize a DOBACK route on a real map colored by stability index.

Input must be a CSV file with columns: timestamp, lat, lon, si (or si_total).

Usage:
  python Scripts/visualize_doback_route.py "Doback-Data/processed data/DOBACK024_20251020.csv"
  python Scripts/visualize_doback_route.py "file1.csv" "file2.csv" "file3.csv"
  python Scripts/visualize_doback_route.py "Doback-Data/processed data/DOBACK024_20251020.csv" --output custom.html
"""

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("folium no instalado. Instala con: pip install folium")


def si_to_color(si_value):
    """
    Convert SI value to color using a gradient from red to green.
    
    Args:
        si_value: Stability index value
    
    Returns:
        Hex color string
    """
    # Clamp SI value to the fixed 0-1 range
    si_clamped = max(0.0, min(1.0, si_value))
    
    # Normalize to 0-1 range
    normalized = si_clamped
    
    # Interpolate from red (255, 0, 0) to green (0, 170, 0)
    # Using HSV-like interpolation through yellow for better visual gradient
    if normalized < 0.5:
        # Red to Yellow (0.0 - 0.5)
        t = normalized * 2  # Scale to 0-1
        r = 255
        g = int(255 * t)
        b = 0
    else:
        # Yellow to Green (0.5 - 1.0)
        t = (normalized - 0.5) * 2  # Scale to 0-1
        r = int(255 * (1 - t))
        g = 255 - int(85 * t)  # Fade from 255 to 170
        b = 0
    
    return f"#{r:02X}{g:02X}{b:02X}"


def load_route_data(csv_path):
    """Load route data from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")
    
    if not csv_path.suffix == ".csv":
        raise ValueError(f"El archivo debe ser CSV: {csv_path}")
    
    print(f"Cargando datos de: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    # Find SI column
    si_col = None
    for candidate in ["si", "si_total", "SI", "si_stab"]:
        if candidate in df.columns:
            si_col = candidate
            break
    
    if si_col is None:
        raise ValueError(f"No se encontró columna de SI. Columnas disponibles: {df.columns.tolist()}")
    
    # Check required columns
    required = ["lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {df.columns.tolist()}")
    
    # Select and clean data
    df = df[["timestamp", "lat", "lon", si_col]].copy()
    df = df.dropna(subset=["lat", "lon", si_col])
    df = df.rename(columns={si_col: "si"})
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        
        # Aggregate by timestamp to avoid duplicates
        df = (
            df.groupby("timestamp", as_index=False)
            .agg({"lat": "mean", "lon": "mean", "si": "mean"})
        )
    
    if df.empty:
        raise ValueError("No hay datos válidos después de limpiar")
    
    print(f"  ✓ {len(df):,} puntos válidos")
    print(f"  ✓ SI rango: {df['si'].min():.3f} - {df['si'].max():.3f}")
    
    return df


def build_map(segments_data, output_path):
    """Build interactive folium map with multiple route segments colored by SI."""
    if not segments_data:
        raise ValueError("No hay datos válidos para visualizar")

    if not HAS_FOLIUM:
        raise RuntimeError("folium no disponible. Instala con: pip install folium")

    print(f"Generando mapa interactivo con {len(segments_data)} segmento(s)...")
    
    # Calculate global stats
    all_lats = []
    all_lons = []
    all_si = []
    total_points = 0
    
    for segment_name, df in segments_data:
        all_lats.extend(df["lat"].values)
        all_lons.extend(df["lon"].values)
        all_si.extend(df["si"].values)
        total_points += len(df)
    
    min_si = min(all_si)
    max_si = max(all_si)
    print(f"  Rango SI (datos): {min_si:.3f} - {max_si:.3f}")
    print(f"  Total de puntos: {total_points:,}")
    print("  Escala de color fija: 0.000 (rojo) - 1.000 (verde)")
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Plot each segment
    for segment_idx, (segment_name, df) in enumerate(segments_data):
        coords = list(zip(df["lat"].values, df["lon"].values))
        si_values = df["si"].values

        # Draw route segments
        for i in range(len(coords) - 1):
            si_value = si_values[i]
            color = si_to_color(si_value)
            folium.PolyLine(
                [coords[i], coords[i + 1]],
                color=color,
                weight=4,
                opacity=0.85,
                popup=f"{segment_name}<br>SI: {si_value:.3f}",
            ).add_to(m)

        # Mark start and end of each segment
        folium.CircleMarker(
            location=[df["lat"].iloc[0], df["lon"].iloc[0]],
            radius=5,
            popup=f"Inicio {segment_name}",
            color="green",
            fill=True,
            fillColor="green",
        ).add_to(m)

        folium.CircleMarker(
            location=[df["lat"].iloc[-1], df["lon"].iloc[-1]],
            radius=5,
            popup=f"Final {segment_name}",
            color="red",
            fill=True,
            fillColor="red",
        ).add_to(m)

    # Create gradient legend with segment info
    segments_info = "<br>".join([f"• {name}: {len(df):,} pts" for name, df in segments_data[:5]])
    if len(segments_data) > 5:
        segments_info += f"<br>• ... y {len(segments_data) - 5} más"
    
    legend_html = f"""
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 240px; height: auto; max-height: 300px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px; border-radius: 5px; overflow-y: auto;">
        <p style="margin: 0 0 8px 0; font-weight: bold;">Índice de Estabilidad (SI)</p>
        <div style="margin: 10px 0;">
            <div style="width: 100%; height: 30px; background: linear-gradient(to right, #FF0000, #FFFF00, #00AA00); border: 1px solid #333;"></div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 10px;">
                <span>Bajo</span>
                <span>Medio</span>
                <span>Alto</span>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 10px;">
            <p style="margin: 3px 0;">Escala fija: 0.000 - 1.000</p>
            <p style="margin: 3px 0;">Min datos: {min_si:.3f}</p>
            <p style="margin: 3px 0;">Max datos: {max_si:.3f}</p>
            <p style="margin: 3px 0;">Total puntos: {total_points:,}</p>
            <p style="margin: 3px 0;">Segmentos: {len(segments_data)}</p>
        </div>
        <div style="margin-top: 10px; font-size: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
            {segments_info}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))


def find_matching_files(base_pattern, search_dir=None):
    """
    Find all CSV files matching a base pattern.
    
    If base_pattern is an existing file, returns [base_pattern].
    Otherwise, searches for files matching the pattern:
    - PATTERN.csv
    - PATTERN_seg*.csv
    """
    base_path = Path(base_pattern)
    
    # If it's an existing file, return it directly
    if base_path.exists() and base_path.is_file():
        return [base_path]
    
    # Determine search directory
    if base_path.parent.exists() and base_path.parent != Path('.'):
        # Pattern includes a directory path
        search_dir = base_path.parent
        base_name = base_path.name
    elif search_dir is not None:
        search_dir = Path(search_dir)
        base_name = base_path.name
    else:
        # No directory in pattern, try to resolve
        if base_path.is_absolute():
            search_dir = base_path.parent
            base_name = base_path.name
        else:
            search_dir = Path.cwd()
            base_name = base_path.name
    
    # Remove .csv extension if present in base_name
    if base_name.endswith('.csv'):
        base_name = base_name[:-4]
    
    # Search for matching files
    matches = []
    
    # Look for exact match: PATTERN.csv
    exact_match = search_dir / f"{base_name}.csv"
    if exact_match.exists():
        matches.append(exact_match)
    
    # Look for segments: PATTERN_seg*.csv
    segment_pattern = f"{base_name}_seg*.csv"
    segment_matches = sorted(search_dir.glob(segment_pattern))
    matches.extend(segment_matches)
    
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza ruta(s) DOBACK con SI en mapa interactivo",
        epilog="Ejemplos:\n"
               "  %(prog)s file1.csv file2.csv\n"
               "  %(prog)s DOBACK024_20250929  # Busca todos los segmentos automáticamente\n"
               "  %(prog)s \"Doback-Data/processed data/DOBACK024_20250929\"",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_files", nargs="+", help="Archivo(s) CSV o nombre base para buscar segmentos automáticamente")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida HTML (por defecto: output/mapa_ruta_si.html)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abrir automáticamente el navegador",
    )

    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    default_output = project_root / "output" / "mapa_ruta_si.html"
    output_path = Path(args.output).resolve() if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("VISUALIZACIÓN DE RUTA(S) DOBACK CON ÍNDICE DE ESTABILIDAD")
    print(f"{'='*70}\n")

    # Expand patterns to actual files
    all_files = []
    for pattern in args.csv_files:
        matches = find_matching_files(pattern)
        if matches:
            all_files.extend(matches)
            if len(matches) > 1:
                print(f"📁 Patrón '{Path(pattern).name}' expandido a {len(matches)} archivos")
        else:
            print(f"⚠ No se encontraron archivos para: {pattern}")
    
    if not all_files:
        print("❌ No se encontraron archivos CSV válidos")
        return
    
    print(f"\nCargando {len(all_files)} archivo(s)...\n")

    # Load all segments
    segments_data = []
    for csv_path in all_files:
        try:
            df = load_route_data(csv_path)
            segments_data.append((csv_path.stem, df))
        except Exception as e:
            print(f"⚠ Error cargando {csv_path.name}: {e}")
            continue
    
    if not segments_data:
        print("❌ No se pudo cargar ningún archivo válido")
        return
    
    build_map(segments_data, output_path)
    
    print(f"\n✓ Mapa guardado en: {output_path}")
    print(f"{'='*70}\n")
    
    # Open in browser automatically
    if not args.no_browser:
        print("Abriendo mapa en navegador...\n")
        webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()

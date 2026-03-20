"""
GPS Map Matching Script
=======================
Ajusta coordenadas GPS a la carretera mÃ¡s probable usando la red viaria.

CaracterÃ­sticas:
  - Carga la red de carreteras local (GraphML) O la descarga automÃ¡ticamente
    de OpenStreetMap para el Ã¡rea de los datos GPS
  - Proyecta cada punto GPS sobre el segmento de carretera mÃ¡s probable,
    teniendo en cuenta distancia perpendicular y alineaciÃ³n de direcciÃ³n
  - Aplica consistencia temporal entre puntos consecutivos
  - Trabaja con todos los ficheros CSV de un directorio o con uno solo

Columnas nuevas / modificadas en el output:
  lat, lon          â†’ coordenadas ajustadas a la carretera
  x_utm, y_utm      â†’ UTM ETRS89 zona 30N ajustadas
  lat_raw, lon_raw  â†’ coordenadas GPS originales
  dist_to_road_m    â†’ distancia (m) entre GPS original y carretera asignada
  road_name         â†’ nombre de la carretera (si existe en OSM)
  road_ref          â†’ referencia (p.ej. "A-6", "M-503")
  highway           â†’ tipo de vÃ­a OSM (motorway, trunk, primaryâ€¦)
  edge_id           â†’ id interno de la arista

Nota: este script escribe en `Doback-Data/map-matched`. Los CSV enriquecidos
con features de terreno se guardan en `Doback-Data/featured` mediante
`src/lidar_stability/lidar/compute_route_terrain_features.py`.

Uso
---
    # Procesar todos los ficheros (descarga automÃ¡tica de red si hace falta)
    python src/lidar_stability/parsers/map_matching.py

    # Procesar un solo fichero
    python src/lidar_stability/parsers/map_matching.py --file Doback-Data/processed-data/DOBACK023_20251012_seg1.csv

    # Usar red local ya descargada (no consultar OSM)
    python src/lidar_stability/parsers/map_matching.py --network output/road_network.graphml

    # Solo ficheros de un camiÃ³n concreto
    python src/lidar_stability/parsers/map_matching.py --glob "DOBACK023_*.csv"

    # Ajustar parÃ¡metros de calidad
    python src/lidar_stability/parsers/map_matching.py --max-dist 100 --dir-weight 0.5

Dependencias: numpy, pandas, scipy, pyproj, tqdm, osmnx
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
BASE            = Path(__file__).resolve().parents[3]
DEFAULT_NETWORK = BASE / "output" / "road_network.graphml"
DEFAULT_INPUT   = BASE / "Doback-Data" / "processed-data"
DEFAULT_OUTPUT  = BASE / "Doback-Data" / "map-matched"
NETWORK_CACHE   = BASE / "output" / "cached_networks"

# ---------------------------------------------------------------------------
# ParÃ¡metros del algoritmo
# ---------------------------------------------------------------------------
DEFAULT_MAX_DIST_M = 150.0   # radio mÃ¡ximo de bÃºsqueda (m)
DEFAULT_DIR_WEIGHT = 0.35    # peso de la alineaciÃ³n de direcciÃ³n [0-1]
SAMPLE_DENSITY     = 0.25    # puntos de Ã­ndice por metro de carretera
NETWORK_BUFFER_DEG = 0.02    # buffer alrededor del bbox GPS para la descarga

# Tipos de vÃ­a OSM que NO son aptos para vehÃ­culos pesados (camiones).
# Se excluyen del Ã­ndice espacial para que el algoritmo no pueda asignar
# ningÃºn punto GPS a estas vÃ­as.
EXCLUDED_HIGHWAY_TYPES = {
    "footway",       # aceras / sendas peatonales
    "path",          # caminos / senderos genÃ©ricos
    "cycleway",      # carriles bici
    "pedestrian",    # zonas peatonales
    "steps",         # escaleras
    "track",         # pistas agrÃ­colas / forestales
    "bridleway",     # caminos de herradura
    "corridor",      # pasillos interiores
    "via_ferrata",   # vÃ­as ferratas
    "proposed",      # vÃ­as en proyecto (no construidas)
    "construction",  # en obras
    "raceway",       # circuitos privados
    "rest_area",     # Ã¡reas de descanso (no carreteras)
    "elevator",      # ascensores
    "bus_guideway",  # carriles exclusivos de bus guiado
}

# --- ParÃ¡metros de suavizado temporal ---
# Ventana de historial: cuÃ¡ntos puntos recientes influyen en la decisiÃ³n
HISTORY_WINDOW     = 4
# Peso de cada punto de la ventana (exponencial, mÃ¡s reciente = mÃ¡s peso)
# factor < 1: el punto j-Ã©simo mÃ¡s antiguo tiene peso decay^j
HISTORY_DECAY      = 0.70
# La arista dominante en la ventana recibe este multiplicador en su score
HISTORY_BONUS      = 0.45
# Para cambiar de arista, la nueva debe ser al menos este ratio MEJOR
# que la arista actual (histÃ©resis). 0.20 = 20% mejor.
SWITCH_THRESHOLD   = 0.4
# Post-procesado: radio (en nÂº de puntos) del filtro de votaciÃ³n de vecinos
SMOOTH_RADIUS      = 2
# Un salto aislado se corrige si dura menos de SMOOTH_MIN_RUN puntos
SMOOTH_MIN_RUN     = 1

# ---------------------------------------------------------------------------
# Transformadores de coordenadas  (lon/lat WGS84 â†” UTM ETRS89 zone 30N)
# ---------------------------------------------------------------------------
_wgs2utm = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
_utm2wgs = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)


def lonlat_to_utm(lon, lat):
    return _wgs2utm.transform(lon, lat)


def utm_to_lonlat(x, y):
    lon, lat = _utm2wgs.transform(x, y)
    return lon, lat


# ---------------------------------------------------------------------------
# Carga de la red viaria
# ---------------------------------------------------------------------------

def _parse_linestring_wkt(wkt: str):
    """'LINESTRING (lon lat, â€¦)' â†’ lista de (lon, lat)"""
    coords_str = re.sub(r"LINESTRING\s*\(|\)", "", wkt.strip())
    coords = []
    for pair in coords_str.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def load_network_from_graphml(graphml_path: str):
    """
    Parsea un fichero GraphML generado por OSMnx (sin necesitar osmnx).
    Devuelve lista de aristas: [{'id', 'coords', 'name', 'ref', 'highway'}, ...]
    """
    NS = "http://graphml.graphdrawing.org/xmlns"

    def tag(name):
        return f"{{{NS}}}{name}"

    tree = ET.parse(graphml_path)
    root = tree.getroot()

    # Mapear key ids â†’ atributos
    key_map = {}
    for key_el in root.findall(tag("key")):
        kid  = key_el.get("id")
        name = key_el.get("attr.name")
        for_ = key_el.get("for")
        key_map[kid] = (name, for_)

    node_keys = {v[0]: k for k, v in key_map.items() if v[1] == "node"}
    edge_keys = {v[0]: k for k, v in key_map.items() if v[1] == "edge"}

    coord_x_kid = node_keys.get("x")
    coord_y_kid = node_keys.get("y")
    geom_kid    = edge_keys.get("geometry")
    name_kid    = edge_keys.get("name")
    ref_kid     = edge_keys.get("ref")
    hw_kid      = edge_keys.get("highway")

    graph_el = root.find(tag("graph"))

    # Nodos
    nodes = {}
    for node_el in graph_el.findall(tag("node")):
        nid  = node_el.get("id")
        xval = yval = None
        for data in node_el.findall(tag("data")):
            k = data.get("key")
            if k == coord_x_kid:
                xval = float(data.text)
            elif k == coord_y_kid:
                yval = float(data.text)
        if xval is not None and yval is not None:
            nodes[nid] = (xval, yval)   # (lon, lat)

    # Aristas
    edges = []
    for edge_el in graph_el.findall(tag("edge")):
        src = edge_el.get("source")
        tgt = edge_el.get("target")
        geom_wkt = name = ref = highway = None
        for data in edge_el.findall(tag("data")):
            k = data.get("key")
            if k == geom_kid:
                geom_wkt = data.text
            elif k == name_kid:
                name = data.text
            elif k == ref_kid:
                ref = data.text
            elif k == hw_kid:
                highway = data.text

        if geom_wkt:
            coords = _parse_linestring_wkt(geom_wkt)
        else:
            p1, p2 = nodes.get(src), nodes.get(tgt)
            if p1 and p2:
                coords = [p1, p2]
            else:
                continue

        if len(coords) < 2:
            continue

        # Descartar vÃ­as no aptas para vehÃ­culos
        hw_val = (highway or "").strip().lower()
        if hw_val in EXCLUDED_HIGHWAY_TYPES:
            continue

        edges.append({
            "id":      len(edges),
            "coords":  coords,
            "name":    name    or "",
            "ref":     ref     or "",
            "highway": highway or "",
        })

    return edges


def load_network_from_osmnx(bbox, network_type="drive", cache_dir=None):
    """
    Descarga la red viaria de OSM para el bbox using osmnx.
    bbox = (min_lat, max_lat, min_lon, max_lon)
    """
    try:
        import osmnx as ox
    except ImportError:
        print("[ERROR] osmnx no estÃ¡ instalado. InstÃ¡lalo con:\n"
              "  python -m pip install osmnx")
        return []

    min_lat, max_lat, min_lon, max_lon = bbox
    print(f"  Descargando de OSM: lat=[{min_lat:.4f},{max_lat:.4f}] "
          f"lon=[{min_lon:.4f},{max_lon:.4f}]â€¦")

    # osmnx 2.0+ expects bbox=(west, south, east, north) = (left, bottom, right, top)
    osm_bbox = (min_lon, min_lat, max_lon, max_lat)

    try:
        G = ox.graph_from_bbox(
            bbox=osm_bbox,
            network_type=network_type,
            simplify=True,
        )
    except Exception as exc:
        print(f"[ERROR] No se pudo descargar la red OSM: {exc}")
        return []

    # Guardar en cachÃ©
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = (f"net_{min_lat:.4f}_{max_lat:.4f}"
                 f"_{min_lon:.4f}_{max_lon:.4f}.graphml")
        try:
            ox.save_graphml(G, filepath=str(cache_dir / fname))
            print(f"  Guardado en cachÃ©: {fname}")
        except Exception:
            pass

    # Convertir grafo osmnx â†’ lista de aristas homogÃ©nea
    edges = []
    nodes_data = {nid: d for nid, d in G.nodes(data=True)}

    for u, v, key, edata in G.edges(keys=True, data=True):
        geom = edata.get("geometry")
        coords = None
        if geom is not None:
            try:
                coords = [(float(c[0]), float(c[1])) for c in geom.coords]
            except Exception:
                coords = None

        if not coords:
            nu, nv = nodes_data.get(u), nodes_data.get(v)
            if nu and nv:
                coords = [(float(nu["x"]), float(nu["y"])),
                          (float(nv["x"]), float(nv["y"]))]
            else:
                continue

        if len(coords) < 2:
            continue

        name    = edata.get("name")    or ""
        ref     = edata.get("ref")     or ""
        highway = edata.get("highway") or ""
        if isinstance(name, list):
            name = ", ".join(name)
        if isinstance(ref, list):
            ref = ", ".join(ref)

        # Descartar vÃ­as no aptas para vehÃ­culos
        hw_val = str(highway).strip().lower()
        if hw_val in EXCLUDED_HIGHWAY_TYPES:
            continue

        edges.append({
            "id":      len(edges),
            "coords":  coords,
            "name":    str(name),
            "ref":     str(ref),
            "highway": str(highway),
        })

    return edges


def _network_covers_bbox(edges, bbox, min_matches=10):
    """True si la red tiene al menos min_matches nodos dentro del bbox."""
    min_lat, max_lat, min_lon, max_lon = bbox
    count = 0
    for e in edges:
        for lon, lat in e["coords"]:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                count += 1
                if count >= min_matches:
                    return True
    return False


def get_network_for_bbox(bbox, local_graphml=None, cache_dir=None, verbose=True):
    """
    1. Intenta la red local (graphml_path)
    2. Busca en cachÃ© de redes descargadas
    3. Descarga de OSM si es necesario
    """
    # 1. Red local
    if local_graphml and Path(local_graphml).is_file():
        if verbose:
            print(f"Cargando red viaria local: {Path(local_graphml).name}")
        edges = load_network_from_graphml(str(local_graphml))
        if _network_covers_bbox(edges, bbox):
            if verbose:
                print(f"  â†’ {len(edges)} aristas, cubre el Ã¡rea de los datos")
            return edges
        else:
            if verbose:
                print("  â†’ La red local no cubre toda el Ã¡rea de los datos GPS")

    # 2. Redes en cachÃ©
    if cache_dir and Path(cache_dir).is_dir():
        for gml in sorted(Path(cache_dir).glob("net_*.graphml")):
            edges = load_network_from_graphml(str(gml))
            if _network_covers_bbox(edges, bbox):
                if verbose:
                    print(f"  â†’ Usando red en cachÃ©: {gml.name} "
                          f"({len(edges)} aristas)")
                return edges

    # 3. Descargar de OSM
    if verbose:
        print("Red local insuficiente â†’ descargando de OpenStreetMapâ€¦")
    buf = NETWORK_BUFFER_DEG
    min_lat, max_lat, min_lon, max_lon = bbox
    # AÃ±adir buffer y pasar en formato (min_lat, max_lat, min_lon, max_lon)
    dl_bbox = (min_lat - buf, max_lat + buf, min_lon - buf, max_lon + buf)
    return load_network_from_osmnx(dl_bbox, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Ãndice espacial
# ---------------------------------------------------------------------------

def build_spatial_index(edges, sample_density=SAMPLE_DENSITY):
    """Muestrea las aristas en UTM y construye un cKDTree."""
    points_utm  = []
    sample_meta = []

    for edge in edges:
        coords = edge["coords"]
        for i in range(len(coords) - 1):
            lon0, lat0 = coords[i]
            lon1, lat1 = coords[i + 1]
            x0, y0 = lonlat_to_utm(lon0, lat0)
            x1, y1 = lonlat_to_utm(lon1, lat1)
            seg_len = np.hypot(x1 - x0, y1 - y0)
            if seg_len < 1e-3:
                continue
            n_samp = max(2, int(np.ceil(seg_len * sample_density)))
            for t in np.linspace(0.0, 1.0, n_samp):
                px = x0 + t * (x1 - x0)
                py = y0 + t * (y1 - y0)
                points_utm.append([px, py])
                sample_meta.append({"edge_idx": edge["id"], "seg_idx": i})

    if not points_utm:
        return None, []

    return cKDTree(np.array(points_utm)), sample_meta


# ---------------------------------------------------------------------------
# Funciones geomÃ©tricas
# ---------------------------------------------------------------------------

def project_point_on_segment(px, py, ax, ay, bx, by):
    """ProyecciÃ³n ortogonal de P sobre el segmento Aâ†’B, sujeta a [0,1]."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return ax, ay, 0.0, np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    qx, qy = ax + t * dx, ay + t * dy
    return qx, qy, t, np.hypot(px - qx, py - qy)


def bearing_deg(dx, dy):
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angular_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ---------------------------------------------------------------------------
# NÃºcleo del map matching
# ---------------------------------------------------------------------------

def _project_on_edge(edge, px, py):
    """Proyecta (px,py) sobre la arista y devuelve la mejor proyecciÃ³n."""
    best = None
    for j in range(len(edge["coords"]) - 1):
        lon0, lat0 = edge["coords"][j]
        lon1, lat1 = edge["coords"][j + 1]
        ax, ay = lonlat_to_utm(lon0, lat0)
        bx, by = lonlat_to_utm(lon1, lat1)
        qx, qy, t, dist = project_point_on_segment(px, py, ax, ay, bx, by)
        if best is None or dist < best["dist"]:
            best = {"qx": qx, "qy": qy, "dist": dist,
                    "dx": bx - ax, "dy": by - ay}
    return best


def _smooth_edge_assignments(edge_ids, radius=SMOOTH_RADIUS,
                              min_run=SMOOTH_MIN_RUN):
    """
    Post-procesado: elimina saltos aislados en la secuencia de asignaciones.

    Para cada punto i, mira los radius vecinos a cada lado. Si la arista
    asignada a i aparece menos veces que otra arista mayoritaria Y el tramo
    continuo en i tiene menos de min_run puntos, reasigna i a la mayorÃ­a.

    Devuelve el array de edge_ids corregido.
    """
    import collections
    n = len(edge_ids)
    result = edge_ids.copy()

    # Calcular longitud del tramo continuo al que pertenece cada punto
    run_len = np.ones(n, dtype=int)
    for i in range(1, n):
        if result[i] == result[i - 1] and result[i] != -1:
            run_len[i] = run_len[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if result[i] == result[i + 1] and result[i] != -1:
            run_len[i] = max(run_len[i], run_len[i + 1])

    changed = True
    passes  = 0
    while changed and passes < 4:
        changed = False
        passes += 1
        for i in range(n):
            if run_len[i] >= min_run:
                continue   # tramo suficientemente largo â†’ no tocar

            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            window = [result[j] for j in range(lo, hi) if result[j] != -1]
            if not window:
                continue

            counter   = collections.Counter(window)
            majority  = counter.most_common(1)[0][0]
            if majority != result[i] and counter[majority] > counter[result[i]]:
                result[i] = majority
                changed = True

        # Recalcular run_len tras cada pasada
        run_len = np.ones(n, dtype=int)
        for i in range(1, n):
            if result[i] == result[i - 1] and result[i] != -1:
                run_len[i] = run_len[i - 1] + 1
        for i in range(n - 2, -1, -1):
            if result[i] == result[i + 1] and result[i] != -1:
                run_len[i] = max(run_len[i], run_len[i + 1])

    return result


def match_track(df, edges, kdtree, sample_meta,
                max_dist_m=DEFAULT_MAX_DIST_M,
                dir_weight=DEFAULT_DIR_WEIGHT):
    """
    Ajusta cada fila de df a la carretera mÃ¡s probable.

    Estrategia de suavizado:
      1. Ventana deslizante (HISTORY_WINDOW puntos): la arista dominante en la
         historia reciente recibe un bonus exponencial por su frecuencia.
      2. HistÃ©resis: para cambiar de arista, la nueva debe superar a la actual
         por al menos SWITCH_THRESHOLD (ratio relativo).
      3. Post-procesado: filtro de votaciÃ³n por vecindad que elimina saltos
         aislados de duraciÃ³n < SMOOTH_MIN_RUN puntos.
    """
    from collections import deque

    lats = df["lat"].values.astype(float)
    lons = df["lon"].values.astype(float)
    n    = len(df)

    xs_utm, ys_utm = lonlat_to_utm(lons, lats)

    # --- DirecciÃ³n de movimiento suavizada (ventana 3) ---
    dx_raw = np.zeros(n)
    dy_raw = np.zeros(n)
    for i in range(1, n):
        dx_raw[i] = xs_utm[i] - xs_utm[i - 1]
        dy_raw[i] = ys_utm[i] - ys_utm[i - 1]
    if n > 1:
        dx_raw[0], dy_raw[0] = dx_raw[1], dy_raw[1]

    # Suavizar direcciÃ³n con media mÃ³vil de Â±1 para reducir ruido GPS
    dx_move = dx_raw.copy()
    dy_move = dy_raw.copy()
    for i in range(1, n - 1):
        dx_move[i] = (dx_raw[i - 1] + dx_raw[i] + dx_raw[i + 1]) / 3.0
        dy_move[i] = (dy_raw[i - 1] + dy_raw[i] + dy_raw[i + 1]) / 3.0
    has_move = np.hypot(dx_move, dy_move) > 0.5

    out_lat       = lats.copy()
    out_lon       = lons.copy()
    out_x_utm     = xs_utm.copy()
    out_y_utm     = ys_utm.copy()
    out_dist      = np.full(n, np.nan)
    out_road_name = [""] * n
    out_road_ref  = [""] * n
    out_highway   = [""] * n
    out_edge_id   = np.full(n, -1, dtype=int)

    # Historial circular de asignaciones recientes
    history: deque = deque(maxlen=HISTORY_WINDOW)
    current_edge   = -1   # arista en la que estamos actualmente

    for i in range(n):
        px, py = xs_utm[i], ys_utm[i]
        idxs = kdtree.query_ball_point([px, py], r=max_dist_m)
        if not idxs:
            history.clear()
            current_edge = -1
            continue

        # --- Calcular peso de historial por arista ---
        # Peso exponencial: el mÃ¡s reciente tiene decay^0=1, el mÃ¡s antiguo decay^(K-1)
        hist_weight: dict = {}
        for age, eidx_h in enumerate(reversed(history)):   # age=0 mÃ¡s reciente
            w = HISTORY_DECAY ** age
            hist_weight[eidx_h] = hist_weight.get(eidx_h, 0.0) + w
        # Normalizar
        total_hw = sum(hist_weight.values()) or 1.0
        hist_weight = {k: v / total_hw for k, v in hist_weight.items()}

        # --- Evaluar candidatos ---
        cands = {}
        seen_eids = set()
        for idx in idxs:
            eidx = sample_meta[idx]["edge_idx"]
            if eidx in seen_eids:
                continue
            seen_eids.add(eidx)
            edge = edges[eidx]

            best = _project_on_edge(edge, px, py)
            if best is None:
                continue

            dist_score = best["dist"] / max_dist_m

            if has_move[i] and np.hypot(best["dx"], best["dy"]) > 0.1:
                mb = bearing_deg(dx_move[i], dy_move[i])
                sb = bearing_deg(best["dx"], best["dy"])
                dir_score = min(angular_diff(mb, sb), 180.0) / 90.0
                dir_score = min(dir_score, 1.0)
            else:
                dir_score = 0.0

            score = dist_score * (1.0 - dir_weight) + dir_score * dir_weight

            # Bonus por presencia en historial reciente
            hw = hist_weight.get(eidx, 0.0)
            score *= (1.0 - HISTORY_BONUS * hw)

            cands[eidx] = {
                "score":   score,
                "dist":    best["dist"],
                "qx":      best["qx"],
                "qy":      best["qy"],
                "name":    edge["name"],
                "ref":     edge["ref"],
                "highway": edge["highway"],
                "eidx":    eidx,
            }

        if not cands:
            history.clear()
            current_edge = -1
            continue

        # Encontrar el candidato con mejor score
        best_cand = min(cands.values(), key=lambda c: c["score"])

        # --- HistÃ©resis: Â¿merece la pena cambiar de arista? ---
        if current_edge != -1 and current_edge in cands:
            score_current = cands[current_edge]["score"]
            score_new     = best_cand["score"]
            # Solo cambiar si la mejora supera SWITCH_THRESHOLD
            if best_cand["eidx"] != current_edge:
                improvement = (score_current - score_new) / (score_current + 1e-9)
                if improvement < SWITCH_THRESHOLD:
                    # Mantener la arista actual
                    best_cand = cands[current_edge]

        winner = best_cand
        qx, qy = winner["qx"], winner["qy"]
        lon_m, lat_m = utm_to_lonlat(qx, qy)

        out_lat[i]       = lat_m
        out_lon[i]       = lon_m
        out_x_utm[i]     = qx
        out_y_utm[i]     = qy
        out_dist[i]      = winner["dist"]
        out_road_name[i] = winner["name"]
        out_road_ref[i]  = winner["ref"]
        out_highway[i]   = winner["highway"]
        out_edge_id[i]   = winner["eidx"]

        current_edge = winner["eidx"]
        history.append(winner["eidx"])

    # --- PASO 2: Suavizado post-procesado por votaciÃ³n de vecinos ---
    smoothed_ids = _smooth_edge_assignments(out_edge_id)

    # Reproyectar los puntos que han cambiado de arista en el suavizado
    for i in range(n):
        new_eid = smoothed_ids[i]
        if new_eid == out_edge_id[i]:
            continue
        if new_eid == -1:
            continue
        edge = edges[new_eid]
        px, py = xs_utm[i], ys_utm[i]
        best = _project_on_edge(edge, px, py)
        if best is None:
            continue
        qx, qy   = best["qx"], best["qy"]
        lon_m, lat_m = utm_to_lonlat(qx, qy)
        out_lat[i]       = lat_m
        out_lon[i]       = lon_m
        out_x_utm[i]     = qx
        out_y_utm[i]     = qy
        out_dist[i]      = best["dist"]
        out_road_name[i] = edge["name"]
        out_road_ref[i]  = edge["ref"]
        out_highway[i]   = edge["highway"]
        out_edge_id[i]   = new_eid

    return (out_lat, out_lon, out_x_utm, out_y_utm,
            out_dist, out_road_name, out_road_ref, out_highway, out_edge_id)


# ---------------------------------------------------------------------------
# Procesamiento masivo
# ---------------------------------------------------------------------------

def collect_csv_files(input_dir, file_pattern, single_file):
    if single_file:
        p = Path(single_file)
        if not p.is_file():
            print(f"[ERROR] Fichero no encontrado: {single_file}")
            sys.exit(1)
        return [p]
    return sorted(Path(input_dir).glob(file_pattern))


def get_gps_bbox(csv_files, sample_n=50):
    """Calcula el bounding box de todos los ficheros CSV."""
    lats, lons = [], []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["lat", "lon"],
                             nrows=sample_n, on_bad_lines="skip")
            df = df.dropna()
            lats.extend(df["lat"].tolist())
            lons.extend(df["lon"].tolist())
        except Exception:
            pass
    if not lats:
        return None
    return (min(lats), max(lats), min(lons), max(lons))


def process_files(csv_files, output_dir, edges, kdtree, sample_meta,
                  max_dist_m, dir_weight):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "error": 0}

    for csv_file in tqdm(csv_files, desc="Ficheros", unit="csv"):
        try:
            df = pd.read_csv(csv_file, on_bad_lines="skip")

            if "lat" not in df.columns or "lon" not in df.columns:
                tqdm.write(f"  [OMITIDO] {csv_file.name}: sin columnas lat/lon")
                stats["skip"] += 1
                continue

            valid = df["lat"].notna() & df["lon"].notna()
            if valid.sum() == 0:
                tqdm.write(f"  [OMITIDO] {csv_file.name}: sin coordenadas vÃ¡lidas")
                stats["skip"] += 1
                continue

            df["lat_raw"] = df["lat"]
            df["lon_raw"] = df["lon"]

            sub = df[valid].copy()
            (lat_m, lon_m, x_m, y_m,
             dist, rname, rref, rhw, eid) = match_track(
                sub, edges, kdtree, sample_meta, max_dist_m, dir_weight
            )

            df.loc[valid, "lat"]            = lat_m
            df.loc[valid, "lon"]            = lon_m
            df.loc[valid, "x_utm"]          = x_m
            df.loc[valid, "y_utm"]          = y_m
            df.loc[valid, "dist_to_road_m"] = dist
            df.loc[valid, "road_name"]      = rname
            df.loc[valid, "road_ref"]       = rref
            df.loc[valid, "highway"]        = rhw
            df.loc[valid, "edge_id"]        = eid

            out_file = output_path / csv_file.name
            df.to_csv(out_file, index=False)
            stats["ok"] += 1

        except Exception as exc:
            tqdm.write(f"  [ERROR] {csv_file.name}: {exc}")
            stats["error"] += 1

    print(f"\nResumen: {stats['ok']} OK  |  {stats['skip']} omitidos  "
          f"|  {stats['error']} errores")
    print(f"Ficheros ajustados guardados en: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ajusta pistas GPS a la red viaria (map matching).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Uso")[1] if "Uso" in __doc__ else "",
    )
    p.add_argument("--input",      default=str(DEFAULT_INPUT),
                   help="Directorio con CSVs de entrada "
                        f"(default: {DEFAULT_INPUT.relative_to(BASE)})")
    p.add_argument("--output",     default=str(DEFAULT_OUTPUT),
                   help="Directorio de salida "
                        f"(default: {DEFAULT_OUTPUT.relative_to(BASE)})")
    p.add_argument("--network",    default=str(DEFAULT_NETWORK),
                   help="GraphML de red viaria local (si no cubre el Ã¡rea, "
                        "se descarga de OSM automÃ¡ticamente)")
    p.add_argument("--file",       default=None,
                   help="Procesar sÃ³lo este fichero (ruta absoluta o relativa)")
    p.add_argument("--glob",       default="*.csv",
                   help="PatrÃ³n glob para filtrar ficheros (default: *.csv)")
    p.add_argument("--max-dist",   type=float, default=DEFAULT_MAX_DIST_M,
                   help=f"Radio mÃ¡x. de bÃºsqueda en metros (default: {DEFAULT_MAX_DIST_M})")
    p.add_argument("--dir-weight", type=float, default=DEFAULT_DIR_WEIGHT,
                   help=f"Peso de la direcciÃ³n de movimiento 0-1 "
                        f"(default: {DEFAULT_DIR_WEIGHT})")
    p.add_argument("--no-cache",   action="store_true",
                   help="No usar ni guardar redes descargadas en cachÃ©")
    return p.parse_args()


def normalize_cli_path(path_value: str | None) -> str | None:
    """
    Normalize CLI path separators so paths copied from Windows/Linux
    can still work on the other platform.
    """
    if path_value is None:
        return None
    normalized = path_value.replace("\\", "/")
    return str(Path(normalized))


def main():
    args = parse_args()

    args.input = normalize_cli_path(args.input)
    args.output = normalize_cli_path(args.output)
    args.network = normalize_cli_path(args.network)
    args.file = normalize_cli_path(args.file)

    # 1. Recopilar ficheros
    csv_files = collect_csv_files(args.input, args.glob, args.file)
    if not csv_files:
        print(f"[AVISO] No se encontraron ficheros CSV en '{args.input}' "
              f"con patrÃ³n '{args.glob}'")
        sys.exit(0)
    print(f"Ficheros a procesar: {len(csv_files)}")

    # 2. Bounding box del GPS
    print("Calculando Ã¡rea de coberturaâ€¦")
    bbox = get_gps_bbox(csv_files)
    if bbox is None:
        print("[ERROR] No se pudieron leer coordenadas GPS.")
        sys.exit(1)
    min_lat, max_lat, min_lon, max_lon = bbox
    print(f"  Ãrea GPS: lat=[{min_lat:.4f}, {max_lat:.4f}]  "
          f"lon=[{min_lon:.4f}, {max_lon:.4f}]")

    # 3. Cargar / descargar red viaria
    cache_dir = None if args.no_cache else str(NETWORK_CACHE)
    edges = get_network_for_bbox(
        bbox          = bbox,
        local_graphml = args.network,
        cache_dir     = cache_dir,
        verbose       = True,
    )
    if not edges:
        print("[ERROR] No se pudo cargar ninguna red viaria.")
        sys.exit(1)
    print(f"  Total aristas en red: {len(edges)}")

    # 4. Ãndice espacial
    print("Construyendo Ã­ndice espacialâ€¦")
    kdtree, sample_meta = build_spatial_index(edges)
    if kdtree is None:
        print("[ERROR] No se pudo construir el Ã­ndice espacial.")
        sys.exit(1)
    print(f"  Ãndice con {len(sample_meta):,} puntos")

    # 5. Ajustar ficheros
    print(f"\nIniciando ajuste GPS â†’ carreteraâ€¦")
    process_files(
        csv_files   = csv_files,
        output_dir  = args.output,
        edges       = edges,
        kdtree      = kdtree,
        sample_meta = sample_meta,
        max_dist_m  = args.max_dist,
        dir_weight  = args.dir_weight,
    )


if __name__ == "__main__":
    main()


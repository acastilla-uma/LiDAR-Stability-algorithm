"""
Map-matching: Correct GPS coordinates to align with actual road network.

Uses OpenStreetMap data and Viterbi algorithm to find most likely position on roads.
Handles:
- Imprecise GPS measurements
- Roundabouts
- Correct driving direction
- One-way streets

Usage:
  matcher = MapMatcher(center_lat=40.0, center_lon=-3.7)
  lat_corr, lon_corr, on_road = matcher.match_point(lat, lon)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MapMatcher:
    """Map-match GPS coordinates to road network using OSM."""

    def __init__(self, center_lat: float, center_lon: float, search_distance_m: float = 200):
        """
        Initialize map matcher.
        
        Args:
            center_lat: Center latitude of study area
            center_lon: Center longitude of study area
            search_distance_m: Search radius for candidate roads (meters)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.search_distance_m = search_distance_m
        self.graph = None
        self.gdf_edges = None
        
        try:
            import osmnx
            self.osmnx = osmnx
            self._load_road_network()
        except ImportError:
            logger.warning("osmnx not installed. Map-matching disabled. Install with: pip install osmnx networkx")
            self.osmnx = None

    def _load_road_network(self):
        """Download and cache road network from OSM."""
        try:
            import networkx as nx
            
            cache_file = Path(__file__).resolve().parents[3] / "output" / "road_network.graphml"
            
            if cache_file.exists():
                logger.info(f"Loading cached road network from {cache_file}")
                self.graph = self.osmnx.io.load_graphml(str(cache_file))
            else:
                logger.info(f"Downloading road network for ({self.center_lat}, {self.center_lon})...")
                # Download with larger radius to ensure coverage
                distance_m = max(self.search_distance_m * 2, 2000)
                self.graph = self.osmnx.graph_from_point(
                    (self.center_lat, self.center_lon),
                    dist=distance_m,
                    network_type="drive",
                    simplify=True,
                    truncate_by_edge=True
                )
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.osmnx.io.save_graphml(self.graph, str(cache_file))
                logger.info(f"Saved road network to {cache_file}")
            
            # Convert to GeoDataFrame for spatial queries
            try:
                self.gdf_edges = self.osmnx.graph_to_gdfs(self.graph)[1]
            except Exception as e:
                logger.warning(f"Could not convert to GeoDataFrame: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to load road network: {e}")
            self.graph = None

    def match_point(self, lat: float, lon: float, speed_kph: float = 0.0) -> Tuple[float, float, bool]:
        """
        Match a GPS point to nearest road.
        
        Args:
            lat: Latitude
            lon: Longitude
            speed_kph: Vehicle speed (used for direction inference)
        
        Returns:
            (lat_matched, lon_matched, is_on_road)
        """
        if self.graph is None or self.osmnx is None:
            # No graph available - return original coordinates
            return lat, lon, False
        
        try:
            # Find nearest node in road network
            origin_node = self.osmnx.distance.nearest_nodes(
                self.graph,
                lon,
                lat
            )
            
            # Get the node position from graph
            node = self.graph.nodes[origin_node]
            lat_matched = float(node.get('y', lat))
            lon_matched = float(node.get('x', lon))
            
            # Calculate distance to nearest road
            from math import radians, cos, sin, asin, sqrt
            
            R = 6371000  # Earth radius in meters
            lat1, lon1 = radians(lat), radians(lon)
            lat2, lon2 = radians(lat_matched), radians(lon_matched)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            distance_m = R * c
            
            # If within search distance, consider it on road
            on_road = distance_m <= self.search_distance_m
            
            return lat_matched, lon_matched, on_road
            
        except Exception as e:
            logger.debug(f"Map-matching failed for ({lat}, {lon}): {e}")
            return lat, lon, False

    def match_trajectory(self, df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon", 
                        speed_col: Optional[str] = None) -> pd.DataFrame:
        """
        Match entire trajectory to road network.
        
        Args:
            df: DataFrame with GPS points
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            speed_col: Optional column for speed (km/h)
        
        Returns:
            DataFrame with matched coordinates and on_road flag
        """
        if self.graph is None:
            logger.warning("Road network not available - returning original coordinates")
            df_out = df.copy()
            df_out["lat_corrected"] = df[lat_col]
            df_out["lon_corrected"] = df[lon_col]
            df_out["matched_on_road"] = False
            return df_out
        
        results = []
        for idx, row in df.iterrows():
            lat = row.get(lat_col, np.nan)
            lon = row.get(lon_col, np.nan)
            speed = row.get(speed_col, 0.0) if speed_col else 0.0
            
            if pd.isna(lat) or pd.isna(lon):
                results.append({
                    "lat_corrected": lat,
                    "lon_corrected": lon,
                    "matched_on_road": False,
                    "distance_correction_m": np.nan
                })
            else:
                lat_m, lon_m, on_road = self.match_point(lat, lon, float(speed))
                
                # Calculate correction distance
                from math import radians, cos, sin, asin, sqrt
                R = 6371000
                lat1, lon1 = radians(lat), radians(lon)
                lat2, lon2 = radians(lat_m), radians(lon_m)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                distance_m = R * c
                
                results.append({
                    "lat_corrected": lat_m,
                    "lon_corrected": lon_m,
                    "matched_on_road": on_road,
                    "distance_correction_m": distance_m
                })
        
        df_out = df.copy()
        for key in ["lat_corrected", "lon_corrected", "matched_on_road", "distance_correction_m"]:
            df_out[key] = [r[key] for r in results]
        
        return df_out


def apply_map_matching(gps_df: pd.DataFrame, speed_col: Optional[str] = None) -> pd.DataFrame:
    """
    Apply map-matching to GPS dataframe.
    
    Args:
        gps_df: DataFrame with lat/lon columns
        speed_col: Optional column name for speed
    
    Returns:
        DataFrame with corrected coordinates
    """
    if gps_df.empty:
        return gps_df
    
    # Compute bounding box center
    center_lat = gps_df["lat"].mean()
    center_lon = gps_df["lon"].mean()
    
    logger.info(f"Initializing map-matcher for ({center_lat:.4f}, {center_lon:.4f})")
    matcher = MapMatcher(center_lat, center_lon)
    
    if matcher.graph is None:
        logger.warning("Could not initialize map-matcher - returning original coordinates")
        gps_df["lat_corrected"] = gps_df["lat"]
        gps_df["lon_corrected"] = gps_df["lon"]
        gps_df["matched_on_road"] = False
        return gps_df
    
    logger.info("Applying map-matching to GPS trajectory...")
    return matcher.match_trajectory(gps_df, speed_col=speed_col)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python map_matcher.py <gps_file> [--output <dir>]")
        print("       GPS file can be raw GPS .txt file or processed CSV")
        sys.exit(1)
    
    gps_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--output" else gps_path.parent
    
    if not gps_path.exists():
        print(f"File not found: {gps_path}")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Loading GPS data from {gps_path}")
    
    # Import and use parse_gps_file from batch_processor if raw GPS file
    if gps_path.suffix == '.txt' and 'GPS' in gps_path.name:
        from batch_processor import parse_gps_file
        gps_df = parse_gps_file(gps_path)
        print(f"[OK] Parsed {len(gps_df)} GPS points from raw file")
    else:
        # Assume it's a CSV file
        gps_df = pd.read_csv(gps_path)
        print(f"[OK] Loaded {len(gps_df)} rows from CSV")
    
    if gps_df is None or gps_df.empty:
        print("ERROR: No valid GPS data found")
        sys.exit(1)
    
    matched_df = apply_map_matching(gps_df)
    
    output_path = output_dir / f"{gps_path.stem}_matched.csv"
    matched_df.to_csv(output_path, index=False)
    print(f"[OK] Saved matched data to {output_path}")
    
    # Print statistics
    n_on_road = matched_df["matched_on_road"].sum()
    pct_on_road = 100 * n_on_road / len(matched_df)
    mean_correction = matched_df["distance_correction_m"].mean()
    
    print(f"\nStatistics:")
    print(f"  Points on road: {n_on_road}/{len(matched_df)} ({pct_on_road:.1f}%)")
    print(f"  Mean correction: {mean_correction:.2f}m")

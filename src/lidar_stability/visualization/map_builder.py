"""
Map Building for Visualization

Creates interactive 2D (Folium) and 3D (Plotly) maps with GPS route,
LiDAR points, and terrain/error statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pyproj import Transformer

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


_WGS84_TO_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_UTM30N_TO_WEB_MERCATOR = Transformer.from_crs("EPSG:25830", "EPSG:3857", always_xy=True)


def _si_to_color(si_value: float) -> str:
    """Convert Stability Index value to a red-yellow-green gradient."""
    si_clamped = max(0.0, min(1.0, si_value))
    if si_clamped < 0.5:
        t = si_clamped * 2
        r = 255
        g = int(255 * t)
        b = 0
    else:
        t = (si_clamped - 0.5) * 2
        r = int(255 * (1 - t))
        g = int(255 - 85 * t)
        b = 0
    return f"#{r:02X}{g:02X}{b:02X}"


def _elevation_to_color(elevation: float, min_elev: float, max_elev: float) -> str:
    """Convert elevation to color gradient (blue=low to red=high)."""
    if max_elev == min_elev:
        norm = 0.5
    else:
        norm = (elevation - min_elev) / (max_elev - min_elev)
    norm = max(0.0, min(1.0, norm))

    if norm < 0.25:
        t = norm * 4
        r, g, b = 0, int(255 * t), 255
    elif norm < 0.5:
        t = (norm - 0.25) * 4
        r, g, b = 0, 255, int(255 * (1 - t))
    elif norm < 0.75:
        t = (norm - 0.5) * 4
        r, g, b = int(255 * t), 255, 0
    else:
        t = (norm - 0.75) * 4
        r, g, b = 255, int(255 * (1 - t)), 0

    return f"#{r:02X}{g:02X}{b:02X}"


def _route_df(segment_data):
    return segment_data.df.sort_values("timestamp").reset_index(drop=True)


def _to_web_mercator_from_wgs84(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 coordinates to Web Mercator meters."""
    x_3857, y_3857 = _WGS84_TO_WEB_MERCATOR.transform(float(lon), float(lat))
    return x_3857, y_3857


def _to_web_mercator_from_utm30(x: float, y: float) -> tuple[float, float]:
    """Convert UTM 30N coordinates to Web Mercator meters."""
    x_3857, y_3857 = _UTM30N_TO_WEB_MERCATOR.transform(float(x), float(y))
    return x_3857, y_3857


def create_segment_visualization_2d(
    segment_data,
    point_cloud_data,
    segment_score,
    output_path: Path,
) -> Path:
    """Create 2D interactive Folium map."""
    if not HAS_FOLIUM:
        raise ImportError("folium required for 2D visualization. Install with: pip install folium")

    logger.info("Creating 2D map for %s", segment_data.segment_id)
    route_df = _route_df(segment_data)

    m = folium.Map(
        location=[segment_data.center_lat, segment_data.center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    route_group = folium.FeatureGroup(name="GPS Route (SI)", show=True)
    for i in range(len(route_df) - 1):
        lat1, lon1 = route_df.iloc[i][["lat", "lon"]]
        lat2, lon2 = route_df.iloc[i + 1][["lat", "lon"]]
        si = route_df.iloc[i]["si"]
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=_si_to_color(si),
            weight=3,
            opacity=0.85,
        ).add_to(route_group)

    start_row = route_df.iloc[0]
    end_row = route_df.iloc[-1]
    start_x, start_y = _to_web_mercator_from_wgs84(start_row["lat"], start_row["lon"])
    end_x, end_y = _to_web_mercator_from_wgs84(end_row["lat"], end_row["lon"])
    folium.CircleMarker(
        location=[start_row["lat"], start_row["lon"]],
        radius=6,
        color="green",
        fill=True,
        fillColor="green",
        popup=f"Route Start<br>EPSG:3857: X={start_x:.2f}, Y={start_y:.2f}",
    ).add_to(route_group)
    folium.CircleMarker(
        location=[end_row["lat"], end_row["lon"]],
        radius=6,
        color="red",
        fill=True,
        fillColor="red",
        popup=f"Route End<br>EPSG:3857: X={end_x:.2f}, Y={end_y:.2f}",
    ).add_to(route_group)
    route_group.add_to(m)

    lidar_group = folium.FeatureGroup(name="LiDAR Points", show=True)
    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    points_xyz = point_cloud_data.points_xyz
    min_elev = points_xyz[:, 2].min()
    max_elev = points_xyz[:, 2].max()

    for x, y, z in points_xyz:
        lon, lat = transformer.transform(float(x), float(y))
        x_3857, y_3857 = _to_web_mercator_from_utm30(float(x), float(y))
        color = _elevation_to_color(float(z), float(min_elev), float(max_elev))
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=0,
            popup=f"Z={z:.2f}m<br>EPSG:3857: X={x_3857:.2f}, Y={y_3857:.2f}",
        ).add_to(lidar_group)
    lidar_group.add_to(m)

    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 320px; height: 220px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px">
    <p style="margin: 0 0 8px 0"><b>Segment Analysis (2D)</b></p>
    <p style="margin: 0 0 8px 0"><b>ID:</b> {segment_id}<br/>
       <b>Device:</b> {device}<br/>
       <b>Date:</b> {date}</p>
    <p style="margin: 0 0 8px 0"><b>Mean Error:</b> {mean_error:.4f}±{std_error:.4f}<br/>
       <b>Impact:</b> <span style="color:{impact_color}"><b>{impact}</b></span></p>
    <p style="margin: 0 0 8px 0"><b>Cloud:</b> raw={raw_points:,}, decimated={decimated_points:,} ({decimation_pct:.1f}%)</p>
    <p style="margin: 0; font-size: 10px; color: gray">Elevation: {z_min:.2f}m - {z_max:.2f}m</p>
    </div>
    """.format(
        segment_id=segment_data.segment_id,
        device=segment_score.device,
        date=segment_score.date,
        mean_error=segment_score.mean_error,
        std_error=segment_score.std_error,
        impact=segment_score.impact_type.upper(),
        impact_color="green" if segment_score.impact_type == "positive" else "red",
        raw_points=point_cloud_data.points_count_raw,
        decimated_points=point_cloud_data.points_count_decimated,
        decimation_pct=point_cloud_data.decimation_ratio * 100,
        z_min=float(min_elev),
        z_max=float(max_elev),
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("2D map saved to: %s", output_path)
    return output_path


def create_segment_visualization_3d(
    segment_data,
    point_cloud_data,
    segment_score,
    output_path: Path,
) -> Path:
    """Create 3D interactive Plotly map."""
    if not HAS_PLOTLY:
        raise ImportError("plotly required for 3D visualization. Install with: pip install plotly")

    logger.info("Creating 3D map for %s", segment_data.segment_id)
    route_df = _route_df(segment_data)
    points_xyz = point_cloud_data.points_xyz

    min_elev = float(points_xyz[:, 2].min())
    max_elev = float(points_xyz[:, 2].max())

    if "z_mean" in route_df.columns:
        route_z = route_df["z_mean"].to_numpy(dtype=float)
        if np.any(~np.isfinite(route_z)):
            route_z = np.where(np.isfinite(route_z), route_z, np.nanmedian(points_xyz[:, 2]))
    elif "alt" in route_df.columns:
        route_z = route_df["alt"].to_numpy(dtype=float)
    else:
        route_z = np.full(len(route_df), np.nanmedian(points_xyz[:, 2]))

    si_route = route_df["si"].to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=[_to_web_mercator_from_utm30(x, y)[0] for x, y in points_xyz[:, :2]],
            y=[_to_web_mercator_from_utm30(x, y)[1] for x, y in points_xyz[:, :2]],
            z=points_xyz[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=points_xyz[:, 2],
                colorscale="Earth",
                cmin=min_elev,
                cmax=max_elev,
                opacity=0.55,
                colorbar=dict(title="Elevation (m)"),
            ),
            name="LiDAR cloud",
            hovertemplate="<b>X</b>=%{x:.2f} m (EPSG:3857)<br><b>Y</b>=%{y:.2f} m (EPSG:3857)<br><b>Z</b>=%{z:.2f}<extra></extra>",
        )
    )

    route_x_3857 = []
    route_y_3857 = []
    for lat, lon in zip(route_df["lat"].to_numpy(dtype=float), route_df["lon"].to_numpy(dtype=float)):
        x_3857, y_3857 = _to_web_mercator_from_wgs84(lat, lon)
        route_x_3857.append(x_3857)
        route_y_3857.append(y_3857)

    fig.add_trace(
        go.Scatter3d(
            x=route_x_3857,
            y=route_y_3857,
            z=route_z,
            mode="lines+markers",
            line=dict(color="rgba(40,40,40,0.6)", width=4),
            marker=dict(
                size=4,
                color=si_route,
                colorscale="RdYlGn",
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(title="SI", x=1.10),
            ),
            name="Route",
            hovertemplate="<b>SI</b>=%{marker.color:.3f}<br><b>X</b>=%{x:.2f} m (EPSG:3857)<br><b>Y</b>=%{y:.2f} m (EPSG:3857)<br><b>Z</b>=%{z:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[route_x_3857[0]],
            y=[route_y_3857[0]],
            z=[route_z[0]],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[route_x_3857[-1]],
            y=[route_y_3857[-1]],
            z=[route_z[-1]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond"),
            name="End",
        )
    )

    fig.update_layout(
        title=(
            f"3D Segment Visualization | {segment_data.segment_id}<br>"
            f"<sub>error={segment_score.mean_error:.4f}±{segment_score.std_error:.4f} | "
            f"cloud={point_cloud_data.points_count_decimated:,} decimated points</sub>"
        ),
        template="plotly_white",
        width=1400,
        height=900,
        scene=dict(
            xaxis_title="Web Mercator X (EPSG:3857, m)",
            yaxis_title="Web Mercator Y (EPSG:3857, m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.95)),
        ),
        legend=dict(x=0.01, y=0.99),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("3D map saved to: %s", output_path)
    return output_path


def create_segment_visualization(
    segment_data,
    point_cloud_data,
    segment_score,
    output_path: Path,
    view_mode: str = "3d",
) -> list[Path]:
    """
    Create selected visualization mode.

    Args:
        view_mode: '3d', '2d', or 'both'

    Returns:
        List of generated output paths
    """
    view_mode = view_mode.lower()
    if view_mode not in {"2d", "3d", "both"}:
        raise ValueError(f"Invalid view_mode '{view_mode}'. Use one of: 2d, 3d, both")

    output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    if view_mode == "2d":
        generated.append(
            create_segment_visualization_2d(
                segment_data=segment_data,
                point_cloud_data=point_cloud_data,
                segment_score=segment_score,
                output_path=output_path,
            )
        )
        return generated

    if view_mode == "3d":
        generated.append(
            create_segment_visualization_3d(
                segment_data=segment_data,
                point_cloud_data=point_cloud_data,
                segment_score=segment_score,
                output_path=output_path,
            )
        )
        return generated

    base = output_path.with_suffix("")
    out_3d = base.parent / f"{base.name}_3d.html"
    out_2d = base.parent / f"{base.name}_2d.html"

    generated.append(
        create_segment_visualization_3d(
            segment_data=segment_data,
            point_cloud_data=point_cloud_data,
            segment_score=segment_score,
            output_path=out_3d,
        )
    )
    generated.append(
        create_segment_visualization_2d(
            segment_data=segment_data,
            point_cloud_data=point_cloud_data,
            segment_score=segment_score,
            output_path=out_2d,
        )
    )
    return generated

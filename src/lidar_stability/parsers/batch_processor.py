"""
Process and unify DOBACK GPS and stability data.

- Cleans GPS data and stability data.
- Matches GPS and stability by timestamp.
- Outputs one folder per matched GPS/ESTABILIDAD pair.

Default input:
  Doback-Data/GPS
  Doback-Data/Stability

Default output:
    Doback-Data/processed-data

Usage:
  python src/lidar_stability/process_doback_routes.py
  python src/lidar_stability/process_doback_routes.py --tolerance-seconds 1.0
"""

import argparse
import copy
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

try:
    from .outlier_tracker import OutlierTracker
except ImportError:
    from outlier_tracker import OutlierTracker


DEFAULT_OUTLIER_FILTER_CONFIG = {
    "enabled": True,
    "tracking": {
        "enabled": True,
        "output_subdir": "outliers",
    },
    "gps": {
        "max_hdop": 15.0,
        "min_fix": 1,
        "min_satellites": 4,
        "jump_threshold_m": 100.0,
        "isolated_neighbor_distance_m": 50.0,
        "isolated_window_size": 2,
        "isolated_min_neighbors": 1,
    },
    "imu": {
        "enabled": True,
        "axes": ["ax", "ay", "az"],
        "window_size": 41,
        "min_window_points": 15,
        "iqr_multiplier": 3.0,
        "mad_multiplier": 6.0,
        "iqr_eps": 1e-6,
        "min_axes_to_flag": 2,
    },
}


def _deep_update_dict(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value


def load_outlier_filter_config():
    """Load defaults and override with config/config.py when available."""
    cfg = copy.deepcopy(DEFAULT_OUTLIER_FILTER_CONFIG)

    project_root = Path(__file__).resolve().parents[3]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from config.config import OUTLIER_FILTER_CONFIG

        if isinstance(OUTLIER_FILTER_CONFIG, dict):
            _deep_update_dict(cfg, OUTLIER_FILTER_CONFIG)
    except Exception:
        pass

    try:
        from config.config import GPS_VALIDATION

        if isinstance(GPS_VALIDATION, dict):
            gps_cfg = cfg.setdefault("gps", {})
            if "max_hdop" in GPS_VALIDATION:
                gps_cfg["max_hdop"] = GPS_VALIDATION["max_hdop"]
            if "min_satellites" in GPS_VALIDATION:
                gps_cfg["min_satellites"] = GPS_VALIDATION["min_satellites"]
    except Exception:
        pass

    return cfg


def _drop_internal_columns(df):
    if df is None or df.empty:
        return df
    internal_cols = [c for c in df.columns if c.startswith("_")]
    if not internal_cols:
        return df
    return df.drop(columns=internal_cols, errors="ignore")


def _apply_filter_with_tracking(
    df,
    mask,
    *,
    tracker,
    source_key,
    stage,
    reason,
    source_file,
    metadata=None,
):
    if df is None or df.empty:
        return df

    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=df.index)
    mask = mask.reindex(df.index).fillna(False).astype(bool)

    before_rows = len(df)
    dropped_rows = int(mask.sum())

    if tracker is not None:
        tracker.log_filter_step(
            source_key=source_key,
            stage=stage,
            filter_name=reason,
            before_rows=before_rows,
            dropped_rows=dropped_rows,
        )

    if dropped_rows <= 0:
        return df

    if tracker is not None:
        tracker.log_drops(
            source_key=source_key,
            stage=stage,
            reason=reason,
            source_file=str(source_file),
            dropped_df=df.loc[mask].copy(),
            metadata=metadata,
        )

    return df.loc[~mask].copy()


def parse_gps_file(gps_path, tracker=None, source_key=None, gps_filter_config=None):
    """Parse and clean one GPS file."""
    source_key = source_key or Path(gps_path).stem
    gps_filter_config = gps_filter_config or {}

    rows = []
    with open(gps_path, "r", encoding="latin-1") as f:
        header = f.readline().strip()
        columns = f.readline().strip()

        for line_no, line in enumerate(f, start=3):
            line = line.strip()
            if not line:
                continue
            if "sin datos GPS" in line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue

            hora_raspberry = parts[0]
            fecha = parts[1]
            hora_gps = parts[2]

            time_match = re.search(r"(\d{1,2}:\d{2}:\d{2})", hora_gps)
            if not time_match:
                continue

            time_str = time_match.group(1)
            try:
                timestamp = datetime.strptime(f"{fecha} {time_str}", "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue

            try:
                lat = float(parts[3])
                lon = float(parts[4])
                alt = float(parts[5])
                hdop = float(parts[6])
                fix = int(parts[7])
                numsats = int(parts[8])
                speed_kmh = float(parts[9])
            except ValueError:
                continue

            rows.append({
                "timestamp": timestamp,
                "hora_raspberry": hora_raspberry,
                "fecha": fecha,
                "hora_gps": hora_gps,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "hdop": hdop,
                "fix": fix,
                "numsats": numsats,
                "speed_kmh": speed_kmh,
                "_raw_line_no": line_no,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Basic filters
    lat_lon_mask = ~df["lat"].between(35, 45) | ~df["lon"].between(-10, 5)
    df = _apply_filter_with_tracking(
        df,
        lat_lon_mask,
        tracker=tracker,
        source_key=source_key,
        stage="gps",
        reason="gps_invalid_lat_lon",
        source_file=gps_path,
    )

    alt_mask = ~df["alt"].between(0, 3000, inclusive="neither")
    df = _apply_filter_with_tracking(
        df,
        alt_mask,
        tracker=tracker,
        source_key=source_key,
        stage="gps",
        reason="gps_invalid_altitude",
        source_file=gps_path,
    )

    speed_mask = (df["speed_kmh"] < 0) | (df["speed_kmh"] > 200)
    df = _apply_filter_with_tracking(
        df,
        speed_mask,
        tracker=tracker,
        source_key=source_key,
        stage="gps",
        reason="gps_invalid_speed",
        source_file=gps_path,
    )

    max_hdop = gps_filter_config.get("max_hdop")
    if max_hdop is not None and "hdop" in df.columns:
        hdop_mask = pd.to_numeric(df["hdop"], errors="coerce") > float(max_hdop)
        df = _apply_filter_with_tracking(
            df,
            hdop_mask,
            tracker=tracker,
            source_key=source_key,
            stage="gps",
            reason=f"gps_hdop_gt_{max_hdop}",
            source_file=gps_path,
        )

    min_fix = gps_filter_config.get("min_fix")
    if min_fix is not None and "fix" in df.columns:
        fix_mask = pd.to_numeric(df["fix"], errors="coerce") < int(min_fix)
        df = _apply_filter_with_tracking(
            df,
            fix_mask,
            tracker=tracker,
            source_key=source_key,
            stage="gps",
            reason=f"gps_fix_lt_{min_fix}",
            source_file=gps_path,
        )

    min_satellites = gps_filter_config.get("min_satellites")
    if min_satellites is not None and "numsats" in df.columns:
        sats_mask = pd.to_numeric(df["numsats"], errors="coerce") < int(min_satellites)
        df = _apply_filter_with_tracking(
            df,
            sats_mask,
            tracker=tracker,
            source_key=source_key,
            stage="gps",
            reason=f"gps_satellites_lt_{min_satellites}",
            source_file=gps_path,
        )

    if df.empty:
        return None

    # Add UTM coordinates for distance calculations
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    x_utm, y_utm = transformer.transform(df["lon"].values, df["lat"].values)
    df["x_utm"] = x_utm
    df["y_utm"] = y_utm

    # Filter jumps between consecutive points
    jump_threshold = float(gps_filter_config.get("jump_threshold_m", 100.0))
    distances = np.sqrt(np.diff(x_utm) ** 2 + np.diff(y_utm) ** 2)
    anomaly_mask = np.zeros(len(df), dtype=bool)
    for i, dist in enumerate(distances):
        if dist > jump_threshold:
            anomaly_mask[i] = True
            anomaly_mask[i + 1] = True

    df = _apply_filter_with_tracking(
        df,
        pd.Series(anomaly_mask, index=df.index),
        tracker=tracker,
        source_key=source_key,
        stage="gps",
        reason=f"gps_jump_gt_{jump_threshold}m",
        source_file=gps_path,
        metadata={"jump_threshold_m": jump_threshold},
    )

    if df.empty:
        return None

    # Remove isolated points with few nearby neighbors
    keep_mask = filter_isolated_points(
        df,
        neighbor_distance=float(gps_filter_config.get("isolated_neighbor_distance_m", 50.0)),
        window_size=int(gps_filter_config.get("isolated_window_size", 2)),
        min_neighbors=int(gps_filter_config.get("isolated_min_neighbors", 1)),
        return_keep_mask=True,
    )
    df = _apply_filter_with_tracking(
        df,
        ~keep_mask,
        tracker=tracker,
        source_key=source_key,
        stage="gps",
        reason="gps_isolated_point",
        source_file=gps_path,
    )

    if df is None or df.empty:
        return None

    return _drop_internal_columns(df)


def filter_isolated_points(
    df,
    neighbor_distance=50,
    window_size=2,
    min_neighbors=1,
    return_keep_mask=False,
):
    """
    Remove points with too few nearby neighbors in a local window.

    A point is removed if it has fewer than min_neighbors within neighbor_distance
    among the previous/next window_size points.
    """
    if df is None or df.empty:
        if return_keep_mask:
            return pd.Series(dtype=bool)
        return df

    x = df["x_utm"].values
    y = df["y_utm"].values
    keep_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        count = 0
        for offset in range(1, window_size + 1):
            if i - offset >= 0:
                dx = x[i] - x[i - offset]
                dy = y[i] - y[i - offset]
                if (dx * dx + dy * dy) ** 0.5 <= neighbor_distance:
                    count += 1
            if i + offset < len(df):
                dx = x[i] - x[i + offset]
                dy = y[i] - y[i + offset]
                if (dx * dx + dy * dy) ** 0.5 <= neighbor_distance:
                    count += 1
            if count >= min_neighbors:
                break
        keep_mask[i] = count >= min_neighbors

    if return_keep_mask:
        return pd.Series(keep_mask, index=df.index)

    return df[keep_mask].copy()


def split_into_segments(df, max_gap_meters=1000):
    """
    Split a GPS dataframe into segments based on large gaps.
    
    Returns:
        List of DataFrames, one per continuous segment.
    """
    if df is None or df.empty:
        return []
    
    # Calculate distances between consecutive points
    x_utm = df["x_utm"].values
    y_utm = df["y_utm"].values
    distances = np.sqrt(np.diff(x_utm) ** 2 + np.diff(y_utm) ** 2)
    
    # Find indices where gap is larger than threshold
    gap_indices = np.where(distances > max_gap_meters)[0]
    
    if len(gap_indices) == 0:
        # No large gaps, return entire dataframe as single segment
        return [df]
    
    # Split dataframe at gap points
    segments = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        # gap_idx is the index in the distances array, which corresponds to
        # the point BEFORE the gap. So we split at gap_idx + 1
        end_idx = gap_idx + 1
        segment = df.iloc[start_idx:end_idx].copy()
        if len(segment) > 10:  # Only keep segments with at least 10 points
            segments.append(segment)
        start_idx = end_idx
    
    # Add the last segment
    last_segment = df.iloc[start_idx:].copy()
    if len(last_segment) > 10:
        segments.append(last_segment)
    
    return segments


def _parse_stability_header(header_line):
    parts = [p.strip() for p in header_line.split(";") if p.strip()]
    if len(parts) < 2:
        return None

    # Example: ESTABILIDAD;20/10/2025 12:14:17;DOBACK023;Sesion:1;
    date_str = parts[1]
    try:
        return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        return None


def _rolling_mad(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = np.median(values)
    return np.median(np.abs(values - median))


def detect_imu_outliers_rolling(df, imu_filter_config):
    """Detect IMU outliers with rolling robust statistics (IQR with MAD fallback)."""
    if df is None or df.empty:
        return pd.Series(dtype=bool), pd.DataFrame(index=df.index if df is not None else None)

    axes = [axis for axis in imu_filter_config.get("axes", ["ax", "ay", "az"]) if axis in df.columns]
    if not axes:
        return pd.Series(False, index=df.index), pd.DataFrame(index=df.index)

    window_size = int(imu_filter_config.get("window_size", 41))
    min_window_points = int(imu_filter_config.get("min_window_points", 15))
    iqr_multiplier = float(imu_filter_config.get("iqr_multiplier", 3.0))
    mad_multiplier = float(imu_filter_config.get("mad_multiplier", 6.0))
    iqr_eps = float(imu_filter_config.get("iqr_eps", 1e-6))

    axis_masks = {}
    for axis in axes:
        series = pd.to_numeric(df[axis], errors="coerce")
        rolling = series.rolling(window=window_size, center=True, min_periods=min_window_points)

        median = rolling.median()
        q1 = rolling.quantile(0.25)
        q3 = rolling.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)
        iqr_mask = (series < lower) | (series > upper)

        mad = rolling.apply(_rolling_mad, raw=True)
        mad_sigma = 1.4826 * mad
        mad_limit = mad_multiplier * mad_sigma
        mad_mask = (series - median).abs() > mad_limit

        use_iqr_mask = (iqr > iqr_eps).fillna(False).astype(bool).to_numpy()
        iqr_values = iqr_mask.fillna(False).astype(bool).to_numpy()
        mad_values = mad_mask.fillna(False).astype(bool).to_numpy()
        axis_values = np.where(use_iqr_mask, iqr_values, mad_values)
        axis_masks[axis] = pd.Series(axis_values, index=df.index, dtype=bool)

    axis_mask_df = pd.DataFrame(axis_masks, index=df.index)
    min_axes_to_flag = int(imu_filter_config.get("min_axes_to_flag", 2))
    flagged_axes_count = axis_mask_df.sum(axis=1)
    combined_mask = flagged_axes_count >= min_axes_to_flag

    return combined_mask.fillna(False), axis_mask_df.fillna(False)


def parse_stability_file(stab_path, tracker=None, source_key=None, imu_filter_config=None):
    """Parse stability file and build timestamp via dynamic interpolation between clock markers."""
    source_key = source_key or Path(stab_path).stem
    imu_filter_config = imu_filter_config or {}

    with open(stab_path, "r", encoding="latin-1") as f:
        header_line = f.readline().strip()
        columns_line = f.readline().strip()

        base_datetime = _parse_stability_header(header_line)

        raw_columns = [c.strip() for c in columns_line.split(";") if c.strip()]
        if not raw_columns:
            return None

        columns = raw_columns
        rows = []
        timestamps = []
        raw_line_numbers = []
        pending_row_indices = []
        current_marker = base_datetime

        def _normalize_marker(marker_time: datetime) -> datetime:
            if current_marker is None:
                return marker_time

            normalized = datetime.combine(current_marker.date(), marker_time.time())
            if normalized < current_marker and (current_marker - normalized) > timedelta(hours=12):
                normalized += timedelta(days=1)
            if normalized <= current_marker:
                normalized = current_marker + timedelta(seconds=1)
            return normalized

        def _finalize_pending(next_marker: datetime | None) -> None:
            nonlocal pending_row_indices, current_marker

            if not pending_row_indices:
                if next_marker is not None:
                    current_marker = next_marker
                return

            anchor = current_marker or base_datetime
            if anchor is None and next_marker is not None:
                anchor = next_marker
            if anchor is None:
                pending_row_indices = []
                return

            if next_marker is not None:
                # Stability clock markers are emitted every second; distribute 1s across N rows.
                step_seconds = 1.0 / len(pending_row_indices)
            else:
                step_seconds = 1.0 / len(pending_row_indices)

            if not np.isfinite(step_seconds) or step_seconds <= 0:
                step_seconds = 0.1

            for i, row_idx in enumerate(pending_row_indices, start=1):
                timestamps[row_idx] = anchor + timedelta(seconds=step_seconds * i)

            if next_marker is not None:
                current_marker = next_marker
            else:
                current_marker = timestamps[pending_row_indices[-1]]

            pending_row_indices = []

        for line_no, line in enumerate(f, start=3):
            line = line.strip()
            if not line:
                continue

            # New metadata header may appear mid-file when a new session starts.
            if line.upper().startswith("ESTABILIDAD"):
                next_base = _parse_stability_header(line)
                if next_base is not None:
                    _finalize_pending(next_base)
                    base_datetime = next_base
                    current_marker = next_base
                continue

            if line.lower().startswith("ax;"):
                continue

            # Time marker like 12:14:18
            if re.match(r"^\d{1,2}:\d{2}:\d{2}$", line):
                try:
                    t = datetime.strptime(line, "%H:%M:%S")
                except ValueError:
                    continue

                if current_marker is not None:
                    marker_dt = datetime.combine(current_marker.date(), t.time())
                elif base_datetime is not None:
                    marker_dt = datetime.combine(base_datetime.date(), t.time())
                else:
                    marker_dt = None

                if marker_dt is not None:
                    marker_dt = _normalize_marker(marker_dt)
                    _finalize_pending(marker_dt)
                continue

            parts = [p.strip() for p in line.split(";")]
            while parts and parts[-1] == "":
                parts.pop()

            if len(parts) < 4:
                continue

            values = []
            numeric_count = 0
            for val in parts:
                try:
                    values.append(float(val))
                    numeric_count += 1
                except ValueError:
                    values.append(np.nan)

            if numeric_count < 4:
                continue

            rows.append(values)
            timestamps.append(None)
            raw_line_numbers.append(line_no)
            pending_row_indices.append(len(rows) - 1)

        _finalize_pending(next_marker=None)

    if not rows:
        return None

    max_cols = max(len(row) for row in rows)
    if max_cols > len(columns):
        columns = columns + [f"col_{i}" for i in range(len(columns), max_cols)]

    normalized = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [np.nan] * (max_cols - len(row))
        normalized.append(row[:max_cols])

    df = pd.DataFrame(normalized, columns=columns[:max_cols])

    df["timestamp"] = timestamps
    df["_raw_line_no"] = raw_line_numbers
    df = df.dropna(subset=["timestamp"]).copy()

    if imu_filter_config.get("enabled", True):
        imu_mask, axis_mask_df = detect_imu_outliers_rolling(df, imu_filter_config)
        if not axis_mask_df.empty:
            flagged_axes_count = axis_mask_df.sum(axis=1).astype(int)
            flagged_axes = axis_mask_df.apply(
                lambda row: "|".join([axis for axis, is_flagged in row.items() if bool(is_flagged)]),
                axis=1,
            )
        else:
            flagged_axes_count = pd.Series(0, index=df.index)
            flagged_axes = pd.Series("", index=df.index)

        df = _apply_filter_with_tracking(
            df,
            imu_mask,
            tracker=tracker,
            source_key=source_key,
            stage="stability",
            reason="imu_rolling_outlier",
            source_file=stab_path,
            metadata={
                "flagged_axes_count": flagged_axes_count,
                "flagged_axes": flagged_axes,
                "imu_window_size": int(imu_filter_config.get("window_size", 41)),
                "imu_iqr_multiplier": float(imu_filter_config.get("iqr_multiplier", 3.0)),
                "imu_mad_multiplier": float(imu_filter_config.get("mad_multiplier", 6.0)),
            },
        )

    if df is None or df.empty:
        return None

    return _drop_internal_columns(df)


def match_by_timestamp(gps_df, stab_df, tolerance_seconds):
    """Match stability to GPS using nearest timestamp with tolerance."""
    if gps_df is None or stab_df is None or gps_df.empty or stab_df.empty:
        return None

    gps_sorted = gps_df.sort_values("timestamp")
    stab_sorted = stab_df.sort_values("timestamp")

    merged = pd.merge_asof(
        stab_sorted,
        gps_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("_stab", "_gps"),
    )

    # Keep only matched rows
    if "lat" in merged.columns:
        merged = merged.dropna(subset=["lat"])

    return merged


def build_pairs(gps_dir, stab_dir):
    """Find matching GPS and stability files by device and date suffix."""
    gps_files = sorted(gps_dir.glob("GPS_DOBACK*_*.txt"))
    stab_files = sorted(stab_dir.glob("ESTABILIDAD_DOBACK*_*.txt"))

    def key_from_name(name, prefix):
        if "realtime" in name.lower():
            return None
        m = re.match(rf"{prefix}_(DOBACK\d+?)_(.+)\.txt$", name)
        if not m:
            return None
        device = m.group(1)
        rest = m.group(2)
        return f"{device}_{rest}"

    gps_map = {}
    for path in gps_files:
        key = key_from_name(path.name, "GPS")
        if key:
            gps_map[key] = path

    stab_map = {}
    for path in stab_files:
        key = key_from_name(path.name, "ESTABILIDAD")
        if key:
            stab_map[key] = path

    common_keys = sorted(set(gps_map.keys()) & set(stab_map.keys()))
    pairs = [(key, gps_map[key], stab_map[key]) for key in common_keys]
    return pairs


def process_all(data_dir, output_dir, tolerance_seconds, max_gap_meters=1000, map_matching=False):
    gps_dir = data_dir / "GPS"
    stab_dir = data_dir / "Stability"
    outlier_cfg = load_outlier_filter_config()
    outlier_enabled = bool(outlier_cfg.get("enabled", True))
    gps_filter_cfg = outlier_cfg.get("gps", {}) if outlier_enabled else {}
    imu_filter_cfg = outlier_cfg.get("imu", {}) if outlier_enabled else {"enabled": False}
    tracking_cfg = outlier_cfg.get("tracking", {})
    tracking_enabled = bool(outlier_enabled and tracking_cfg.get("enabled", True))
    outlier_output_dir = output_dir / tracking_cfg.get("output_subdir", "outliers")

    pairs = build_pairs(gps_dir, stab_dir)
    if not pairs:
        print("No matching GPS/ESTABILIDAD pairs found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Import map_matcher only if needed
    if map_matching:
        try:
            try:
                from . import map_matcher
            except ImportError:
                # When run as script directly (not as module)
                import map_matcher
            print("[OK] Map-matching enabled - GPS will be corrected to road network")
        except ImportError:
            print("[WARN] map_matcher module not found. Proceeding without map-matching.")
            map_matching = False

    report_lines = []
    report_lines.append("DOBACK ROUTE PROCESSING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Max gap for splitting: {max_gap_meters}m")
    report_lines.append(
        "Outlier filtering: "
        + ("ENABLED" if outlier_enabled else "DISABLED")
        + " (IMU rolling windows + GPS quality checks)"
    )
    if map_matching:
        report_lines.append("Map-matching: ENABLED (GPS corrected to road network)")
    report_lines.append("")

    for key, gps_path, stab_path in pairs:
        tracker = OutlierTracker(enabled=tracking_enabled)
        gps_df = parse_gps_file(
            gps_path,
            tracker=tracker,
            source_key=key,
            gps_filter_config=gps_filter_cfg,
        )
        stab_df = parse_stability_file(
            stab_path,
            tracker=tracker,
            source_key=key,
            imu_filter_config=imu_filter_cfg,
        )

        if tracking_enabled:
            dropped_path, summary_path = tracker.export(outlier_output_dir, key)
            report_lines.append(
                f"{key}: audit outliers -> {dropped_path.relative_to(output_dir)} | {summary_path.relative_to(output_dir)}"
            )

        # Apply map-matching if enabled  
        if map_matching and gps_df is not None and not gps_df.empty:
            try:
                gps_df = map_matcher.apply_map_matching(gps_df)
                # Use corrected coordinates in matching
                gps_df["lat"] = gps_df.get("lat_corrected", gps_df["lat"])
                gps_df["lon"] = gps_df.get("lon_corrected", gps_df["lon"])
            except Exception as e:
                print(f"âš  Map-matching failed for {key}: {e}. Using original GPS.")

        # Split GPS data into segments based on large gaps
        gps_segments = split_into_segments(gps_df, max_gap_meters)
        
        if not gps_segments:
            report_lines.append(f"{key}: no valid GPS segments")
            continue
        
        if len(gps_segments) == 1:
            # Single continuous route
            merged = match_by_timestamp(gps_segments[0], stab_df, tolerance_seconds)
            if merged is None or merged.empty:
                report_lines.append(f"{key}: no matched rows")
                continue
            
            # Skip segments with too few points
            if len(merged) < 10:
                report_lines.append(f"{key}: solo {len(merged)} puntos (descartado, mÃ­nimo 10)")
                continue

            output_file = output_dir / f"{key}.csv"
            merged.to_csv(output_file, index=False)
            report_lines.append(f"{key}: {len(merged):,} rows -> {output_file.name}")
        else:
            # Multiple segments due to gaps
            segment_count = 0
            skipped_count = 0
            for seg_idx, gps_segment in enumerate(gps_segments, 1):
                merged = match_by_timestamp(gps_segment, stab_df, tolerance_seconds)
                if merged is None or merged.empty:
                    continue
                
                # Skip segments with too few points
                if len(merged) < 10:
                    skipped_count += 1
                    continue
                
                segment_count += 1
                output_file = output_dir / f"{key}_seg{seg_idx}.csv"
                merged.to_csv(output_file, index=False)
                report_lines.append(f"{key}_seg{seg_idx}: {len(merged):,} rows -> {output_file.name}")
            
            if segment_count == 0:
                report_lines.append(f"{key}: {len(gps_segments)} segments found but no matched rows")
            else:
                skip_msg = f", {skipped_count} descartados (<10 pts)" if skipped_count > 0 else ""
                report_lines.append(f"{key}: {len(gps_segments)} segments total, {segment_count} guardados{skip_msg}")

    report_path = output_dir / "processing_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Done. Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Clean and unify DOBACK data")
    parser.add_argument(
        "--data-dir",
        default="Doback-Data",
        help="Base directory with GPS and Stability folders",
    )
    parser.add_argument(
        "--output-dir",
        default="Doback-Data/processed-data",
        help="Output directory for processed routes",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=1.0,
        help="Max time difference for matching GPS to stability",
    )
    parser.add_argument(
        "--max-gap-meters",
        type=float,
        default=100,
        help="Maximum gap distance (meters) before splitting into separate routes",
    )
    parser.add_argument(
        "--map-matching",
        action="store_true",
        help="Enable GPS map-matching to align with road network (requires osmnx)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    data_dir = (project_root / args.data_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    process_all(data_dir, output_dir, args.tolerance_seconds, args.max_gap_meters, args.map_matching)


if __name__ == "__main__":
    main()


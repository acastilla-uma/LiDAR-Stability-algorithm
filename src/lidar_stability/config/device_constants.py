"""Helpers to attach literal device constants from the YAML registry."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lidar_stability.config.device_registry import get_registry

LITERAL_DEVICE_CONSTANT_COLUMNS = (
    "k1",
    "k2",
    "k4_mm",
    "d1_m",
    "coeff",
    "s_mm",
    "alphav",
)


def resolve_device_config_from_filename(filename: str) -> tuple[str, dict[str, Any]]:
    """Resolve a registry config from any filename containing a DOBACK token."""
    registry = get_registry()
    device_id = registry.get_device_from_filename(filename)
    if device_id is None:
        raise ValueError(f"Could not extract DOBACK device ID from filename: {filename}")

    config = registry.get_config(device_id)
    return device_id, config


def extract_literal_device_constants_from_filename(filename: str) -> dict[str, Any]:
    """Return the literal stability constants defined in the YAML for a file."""
    device_id, config = resolve_device_config_from_filename(filename)
    stability_model = config.get("stability_model", {})

    missing = [name for name in LITERAL_DEVICE_CONSTANT_COLUMNS if name not in stability_model]
    if missing:
        raise ValueError(
            f"Missing required literal constants for DOBACK{device_id} in registry: {missing}"
        )

    return {name: stability_model[name] for name in LITERAL_DEVICE_CONSTANT_COLUMNS}


def assign_literal_device_constants(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Assign literal device constants to every row of a dataframe."""
    constants = extract_literal_device_constants_from_filename(filename)
    for column, value in constants.items():
        df[column] = value
    return df

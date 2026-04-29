from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

scripts_path = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_path))

from lidar_stability.config.device_constants import (
    LITERAL_DEVICE_CONSTANT_COLUMNS,
    extract_literal_device_constants_from_filename,
)
from lidar_stability.lidar.compute_route_terrain_features import enrich_route_with_terrain_features
from lidar_stability.ml.feature_engineering import DEFAULT_FEATURE_COLUMNS, build_w_training_dataset


EXPECTED_DEVICE_CONSTANTS = {
    "DOBACK023_20251012.csv": {
        "k1": 1.15,
        "k2": 2.05,
        "k4_mm": 1100,
        "d1_m": 2.03,
        "coeff": 1.89,
        "s_mm": 2450,
        "alphav": 54,
    },
    "DOBACK024_20251012.csv": {
        "k1": 1.15,
        "k2": 2.0,
        "k4_mm": 2500,
        "d1_m": 1.88,
        "coeff": 5.00,
        "s_mm": 2215,
        "alphav": 64,
    },
    "DOBACK027_20251020.csv": {
        "k1": 1.50,
        "k2": 1.50,
        "k4_mm": 1100,
        "d1_m": 0.96,
        "coeff": 2.13,
        "s_mm": 1880,
        "alphav": 45,
    },
    "DOBACK028_20251008_seg17.csv": {
        "k1": 1.15,
        "k2": 2.05,
        "k4_mm": 1100,
        "d1_m": 1.96,
        "coeff": 1.86,
        "s_mm": 2200,
        "alphav": 57,
    },
}


@pytest.mark.parametrize("filename,expected", EXPECTED_DEVICE_CONSTANTS.items())
def test_extract_literal_device_constants_from_filename_returns_yaml_values(filename, expected):
    assert extract_literal_device_constants_from_filename(filename) == expected


def test_enrich_route_with_terrain_features_assigns_literal_constants_without_lidar(tmp_path):
    mapmatch_path = tmp_path / "DOBACK027_20251020.csv"
    output_path = tmp_path / "featured.csv"

    pd.DataFrame(
        {
            "x_utm": [447666.88, 447667.25, 447667.62],
            "y_utm": [4487315.05, 4487315.41, 4487315.77],
            "gy": [1.0, 2.0, 3.0],
        }
    ).to_csv(mapmatch_path, index=False)

    result = enrich_route_with_terrain_features(
        mapmatch_path=str(mapmatch_path),
        laz_dir=str(tmp_path / "missing-laz"),
        output_path=str(output_path),
    )

    saved = pd.read_csv(output_path)
    expected = EXPECTED_DEVICE_CONSTANTS[mapmatch_path.name]

    for column, value in expected.items():
        assert column in result.columns
        assert column in saved.columns
        assert result[column].nunique() == 1
        assert saved[column].nunique() == 1
        assert saved[column].iloc[0] == pytest.approx(value)


def test_build_w_training_dataset_only_uses_literal_constants_when_explicit():
    n = 40
    idx = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "roll": np.sin(idx / 10.0),
            "pitch": np.cos(idx / 11.0),
            "ax": 10.0 + idx,
            "ay": 20.0 + idx,
            "az": 30.0 + idx,
            "speed_kmh": 40.0 + idx,
            "phi_lidar": 0.1 + idx / 1000.0,
            "tri": 0.2 + idx / 1000.0,
            "ruggedness": 0.3 + idx / 1000.0,
            "gy": 50.0 + idx,
            "k1": 1.15,
            "k2": 2.0,
            "k4_mm": 2500,
            "d1_m": 1.88,
            "coeff": 5.0,
            "s_mm": 2215,
            "alphav": 64,
        }
    )

    _, _, used_default, _ = build_w_training_dataset(df)
    assert not set(LITERAL_DEVICE_CONSTANT_COLUMNS).intersection(used_default)

    explicit_columns = list(DEFAULT_FEATURE_COLUMNS) + list(LITERAL_DEVICE_CONSTANT_COLUMNS)
    _, _, used_explicit, _ = build_w_training_dataset(df, feature_columns=explicit_columns)
    assert set(LITERAL_DEVICE_CONSTANT_COLUMNS).issubset(used_explicit)

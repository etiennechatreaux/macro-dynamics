"""Test that RollingZScoreTransformer uses shift(1) to prevent leakage."""

import numpy as np
import pandas as pd
import pytest

from macrostate.features.transformers import RollingZScoreTransformer


def test_zscore_uses_only_past_data():
    """Verify z-score at time t uses only data from t-1 and earlier."""
    # Create simple test data
    dates = pd.date_range("2020-01-01", periods=10, freq="ME")
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=dates)

    transformer = RollingZScoreTransformer(
        columns=["value"], window=3, min_periods=2, suffix="_Z"
    )
    result = transformer.transform(df)

    # At t=2 (value=3), z-score should use mean/std of [1,2] (shifted values)
    # Not [1,2,3]
    mean_past = np.mean([1, 2])  # 1.5
    std_past = np.std([1, 2], ddof=1)  # ~0.707
    expected_z = (3 - mean_past) / std_past

    # Index 2 is the third row
    actual_z = result["value_Z"].iloc[2]
    assert np.isclose(actual_z, expected_z, rtol=0.01), (
        f"Z-score should be {expected_z:.4f} but got {actual_z:.4f}"
    )


def test_zscore_first_values_are_nan():
    """First values should be NaN due to min_periods and shift."""
    dates = pd.date_range("2020-01-01", periods=5, freq="ME")
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

    transformer = RollingZScoreTransformer(
        columns=["value"], window=3, min_periods=2, suffix="_Z"
    )
    result = transformer.transform(df)

    # First two values should be NaN (shift(1) + min_periods=2)
    assert pd.isna(result["value_Z"].iloc[0])
    assert pd.isna(result["value_Z"].iloc[1])


def test_zscore_does_not_include_current_value():
    """Ensure current value is NOT included in rolling stats."""
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    # Spike at position 3
    df = pd.DataFrame({"value": [1, 1, 1, 100, 1, 1]}, index=dates)

    transformer = RollingZScoreTransformer(
        columns=["value"], window=3, min_periods=2, suffix="_Z"
    )
    result = transformer.transform(df)

    # At t=3 (value=100), mean/std should be computed on [1,1,1] from positions 0,1,2
    # If current value was included, mean would be much higher
    z_at_spike = result["value_Z"].iloc[3]

    # Z-score should be very high (100 is far from mean of 1)
    assert z_at_spike > 50, f"Z-score at spike should be very high, got {z_at_spike}"


def test_drawdown_uses_past_only():
    """Test DrawdownFromCumRetTransformer uses shift(1)."""
    from macrostate.features.transformers import (
        CumulativeReturnTransformer,
        DrawdownFromCumRetTransformer,
    )

    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    df = pd.DataFrame({"SPX_RET_1M": [0.1, 0.1, 0.1, -0.2, 0.05, 0.05]}, index=dates)

    cum_trans = CumulativeReturnTransformer(ret_col="SPX_RET_1M")
    dd_trans = DrawdownFromCumRetTransformer(cum_col="SPX_CUM", window=12)

    df = cum_trans.transform(df)
    df = dd_trans.transform(df)

    # Drawdown at first row should be NaN (no past data)
    assert pd.isna(df["SPX_DD_12M"].iloc[0])


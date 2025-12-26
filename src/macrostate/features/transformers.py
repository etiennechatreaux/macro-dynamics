from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasTransformer(BaseEstimator, TransformerMixin):
    """Base transformer that works with pandas DataFrames."""

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class YieldCurveSlopeTransformer(PandasTransformer):
    """Create yield curve slope: YC_SLOPE = US10Y - US2Y."""

    def __init__(self, long_rate: str = "US10Y", short_rate: str = "US2Y"):
        self.long_rate = long_rate
        self.short_rate = short_rate

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["YC_SLOPE"] = X[self.long_rate] - X[self.short_rate]
        return X


class DrawdownTransformer(PandasTransformer):
    """Compute drawdown from rolling max (past-only to avoid leakage).
    
    For use when price data is available. Computes: price / rolling_max - 1
    Uses shift(1) to ensure only past data is used.
    """

    def __init__(self, price_col: str, window: int = 12, output_col: str | None = None):
        self.price_col = price_col
        self.window = window
        self.output_col = output_col or f"{price_col}_DD_{window}M"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # shift(1) ensures we use only past data (up to t-1)
        rolling_max = X[self.price_col].shift(1).rolling(window=self.window, min_periods=1).max()
        X[self.output_col] = X[self.price_col] / rolling_max - 1
        return X


class DiffTransformer(PandasTransformer):
    """Compute differences (momentum) for specified columns."""

    def __init__(self, columns: list[str], periods: list[int] | None = None):
        self.columns = columns
        self.periods = periods or [1, 6]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            for period in self.periods:
                X[f"{col}_D{period}M"] = X[col].diff(period)
        return X


class RollingZScoreTransformer(PandasTransformer):
    """Compute rolling z-scores using past-only data (no leakage).
    
    Critical: Uses shift(1) before computing rolling mean/std to ensure
    the z-score at time t only uses data from t-1 and earlier.
    """

    def __init__(
        self,
        columns: list[str],
        window: int = 60,
        min_periods: int = 24,
        suffix: str = "_Z",
    ):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.suffix = suffix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            # CRITICAL: shift(1) to avoid leakage - only use past data
            shifted = X[col].shift(1)
            rolling_mean = shifted.rolling(window=self.window, min_periods=self.min_periods).mean()
            rolling_std = shifted.rolling(window=self.window, min_periods=self.min_periods).std()
            # Use current value minus past mean/std
            X[f"{col}{self.suffix}"] = (X[col] - rolling_mean) / rolling_std
        return X


class SignFlipTransformer(PandasTransformer):
    """Flip sign of specified columns (for indicators where higher = worse)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = -X[col]
        return X


class ColumnRenamer(PandasTransformer):
    """Rename columns using a mapping dict."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.rename(columns=self.mapping)


class ColumnSelector(PandasTransformer):
    """Select specific columns from DataFrame."""

    def __init__(self, columns: list[str] | None = None, pattern: str | None = None):
        self.columns = columns
        self.pattern = pattern

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            existing = [c for c in self.columns if c in X.columns]
            return X[existing]
        if self.pattern:
            cols = X.filter(regex=self.pattern).columns.tolist()
            return X[cols]
        return X


class DropNaTransformer(PandasTransformer):
    """Drop rows with NaN values (use after all feature engineering)."""

    def __init__(self, subset: list[str] | None = None):
        self.subset = subset

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna(subset=self.subset)


class CumulativeReturnTransformer(PandasTransformer):
    """Compute cumulative returns from log returns for drawdown calculation."""

    def __init__(self, ret_col: str = "SPX_RET_1M", output_col: str = "SPX_CUM"):
        self.ret_col = ret_col
        self.output_col = output_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Cumsum of log returns = log(cumulative price ratio)
        X[self.output_col] = X[self.ret_col].cumsum()
        return X


class DrawdownFromCumRetTransformer(PandasTransformer):
    """Compute drawdown from cumulative log returns (past-only)."""

    def __init__(self, cum_col: str = "SPX_CUM", window: int = 12, output_col: str = "SPX_DD_12M"):
        self.cum_col = cum_col
        self.window = window
        self.output_col = output_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # shift(1) for past-only, then rolling max
        shifted = X[self.cum_col].shift(1)
        rolling_max = shifted.rolling(window=self.window, min_periods=1).max()
        # Drawdown in log space
        X[self.output_col] = X[self.cum_col] - rolling_max
        return X


from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    # Data paths
    raw_data_path: Path = Path("data/raw/raw_dataset.xlsx")
    cleaned_parquet_path: Path = Path("data/cleaned/monthly.parquet")
    features_dir: Path = Path("data/features")
    reports_dir: Path = Path("reports")

    # Required columns in raw data
    required_columns: list[str] = field(
        default_factory=lambda: [
            "Date",
            "US10Y",
            "US2Y",
            "HY_OAS",
            "IG_OAS",
            "Inflation (expectation)",
            "PMI Gap",
            "Unemployment",
            "Volatilité",
            "S&P500",
            "Credit Spread",
            "Confidence",
        ]
    )

    # Column mapping for cleaner names
    column_rename: dict[str, str] = field(
        default_factory=lambda: {
            "Inflation (expectation)": "INFLATION_EXP",
            "PMI Gap": "PMI_GAP",
            "Volatilité": "VIX",
            "S&P500": "SPX_RET_1M",
            "Credit Spread": "CREDIT_SPREAD",
        }
    )

    # Cleaning parameters
    ffill_max_gap: int = 2
    drop_initial_na: bool = True

    # Z-score parameters
    zscore_window: int = 60
    zscore_min_periods: int = 24

    # Columns for z-score transformation (after renaming)
    zscore_columns: list[str] = field(
        default_factory=lambda: [
            "US10Y",
            "US2Y",
            "HY_OAS",
            "IG_OAS",
            "INFLATION_EXP",
            "PMI_GAP",
            "Unemployment",
            "VIX",
            "CREDIT_SPREAD",
            "Confidence",
            "YC_SLOPE",
        ]
    )

    # Columns for diff transformation
    diff_columns: list[str] = field(
        default_factory=lambda: [
            "US10Y",
            "HY_OAS",
            "INFLATION_EXP",
            "PMI_GAP",
            "Unemployment",
            "VIX",
            "Confidence",
        ]
    )

    # Columns to flip sign (higher = worse)
    sign_flip_columns: list[str] = field(
        default_factory=lambda: ["Unemployment_Z"]
    )

    # Date filter
    asof_date: str | None = None

    def get_features_path(self, recipe: str) -> Path:
        return self.features_dir / f"{recipe}.parquet"


# Available recipes
RecipeType = Literal["baseline_z", "z_plus_momentum", "changes_only", "levels_only"]


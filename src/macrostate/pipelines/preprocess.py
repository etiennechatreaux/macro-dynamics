from sklearn.pipeline import Pipeline

from macrostate.config.settings import PreprocessConfig, RecipeType
from macrostate.features.transformers import (
    YieldCurveSlopeTransformer,
    DiffTransformer,
    RollingZScoreTransformer,
    SignFlipTransformer,
    DropNaTransformer,
    CumulativeReturnTransformer,
    DrawdownFromCumRetTransformer,
)

AVAILABLE_RECIPES = ["baseline_z", "z_plus_momentum", "changes_only", "levels_only"]


def build_preprocessing_pipeline(recipe: RecipeType, cfg: PreprocessConfig) -> Pipeline:
    """Build a preprocessing pipeline based on recipe name.
    
    Recipes:
    - baseline_z: YC_SLOPE + rolling z-scores on levels only
    - z_plus_momentum: baseline_z + Δ1M/Δ6M + SPX drawdown
    - changes_only: only diffs/momentum, no level z-scores
    - levels_only: only level z-scores, no diffs
    """
    if recipe not in AVAILABLE_RECIPES:
        raise ValueError(f"Unknown recipe '{recipe}'. Available: {AVAILABLE_RECIPES}")

    steps: list[tuple[str, any]] = []

    # Common: add yield curve slope
    steps.append(("yc_slope", YieldCurveSlopeTransformer()))

    if recipe == "baseline_z":
        steps.extend(_baseline_z_steps(cfg))

    elif recipe == "z_plus_momentum":
        steps.extend(_z_plus_momentum_steps(cfg))

    elif recipe == "changes_only":
        steps.extend(_changes_only_steps(cfg))

    elif recipe == "levels_only":
        steps.extend(_levels_only_steps(cfg))

    # Always drop NaN at the end
    steps.append(("drop_na", DropNaTransformer()))

    return Pipeline(steps)


def _baseline_z_steps(cfg: PreprocessConfig) -> list[tuple[str, any]]:
    """Baseline: YC_SLOPE + z-scores on levels only."""
    return [
        (
            "zscore",
            RollingZScoreTransformer(
                columns=cfg.zscore_columns,
                window=cfg.zscore_window,
                min_periods=cfg.zscore_min_periods,
            ),
        ),
        ("sign_flip", SignFlipTransformer(columns=cfg.sign_flip_columns)),
    ]


def _z_plus_momentum_steps(cfg: PreprocessConfig) -> list[tuple[str, any]]:
    """Z-scores + momentum (diffs) + SPX drawdown."""
    return [
        # Cumulative returns for drawdown calculation
        ("cum_ret", CumulativeReturnTransformer(ret_col="SPX_RET_1M")),
        # Drawdown from cumulative returns
        ("drawdown", DrawdownFromCumRetTransformer(cum_col="SPX_CUM", window=12)),
        # Diffs (momentum)
        ("diff", DiffTransformer(columns=cfg.diff_columns, periods=[1, 6])),
        # Z-scores
        (
            "zscore",
            RollingZScoreTransformer(
                columns=cfg.zscore_columns,
                window=cfg.zscore_window,
                min_periods=cfg.zscore_min_periods,
            ),
        ),
        ("sign_flip", SignFlipTransformer(columns=cfg.sign_flip_columns)),
    ]


def _changes_only_steps(cfg: PreprocessConfig) -> list[tuple[str, any]]:
    """Only diffs/momentum, no level z-scores."""
    return [
        ("cum_ret", CumulativeReturnTransformer(ret_col="SPX_RET_1M")),
        ("drawdown", DrawdownFromCumRetTransformer(cum_col="SPX_CUM", window=12)),
        ("diff", DiffTransformer(columns=cfg.diff_columns, periods=[1, 6])),
    ]


def _levels_only_steps(cfg: PreprocessConfig) -> list[tuple[str, any]]:
    """Only level z-scores, no diffs."""
    return [
        (
            "zscore",
            RollingZScoreTransformer(
                columns=cfg.zscore_columns,
                window=cfg.zscore_window,
                min_periods=cfg.zscore_min_periods,
            ),
        ),
        ("sign_flip", SignFlipTransformer(columns=cfg.sign_flip_columns)),
    ]


from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from macrostate.config.settings import PreprocessConfig
from macrostate.data.io import (
    load_raw_data,
    clean_dataframe,
    save_parquet,
    generate_data_quality_report,
)
from macrostate.data.validation import validate_dataframe, check_monthly_frequency
from macrostate.pipelines.preprocess import build_preprocessing_pipeline, AVAILABLE_RECIPES
from macrostate.utils.logging import get_logger
from macrostate.utils.paths import get_project_root

app = typer.Typer(name="macrostate", help="Macro regime detection preprocessing CLI")
console = Console()
logger = get_logger("cli")


@app.command()
def preprocess(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to raw data file"),
    ] = None,
    recipe: Annotated[
        str,
        typer.Option("--recipe", "-r", help=f"Preprocessing recipe: {AVAILABLE_RECIPES}"),
    ] = "z_plus_momentum",
    asof: Annotated[
        str,
        typer.Option("--asof", help="Filter data up to this date (YYYY-MM-DD)"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Output directory for features"),
    ] = None,
) -> None:
    """Run preprocessing pipeline on macro data."""
    root = get_project_root()
    cfg = PreprocessConfig()

    if input_path is None:
        input_path = root / cfg.raw_data_path
    if output_dir:
        cfg.features_dir = output_dir
    if asof:
        cfg.asof_date = asof

    if recipe not in AVAILABLE_RECIPES:
        console.print(f"[red]Error: Unknown recipe '{recipe}'[/red]")
        console.print(f"Available recipes: {AVAILABLE_RECIPES}")
        raise typer.Exit(1)

    console.print(f"[bold blue]Macrostate Preprocessing Pipeline[/bold blue]")
    console.print(f"Recipe: [green]{recipe}[/green]")
    console.print(f"Input: {input_path}")

    # Load and validate
    df_raw = load_raw_data(input_path, cfg)
    validate_dataframe(df_raw, cfg)
    warnings = check_monthly_frequency(df_raw)
    for w in warnings:
        console.print(f"[yellow]Warning: {w}[/yellow]")

    # Clean
    df_clean = clean_dataframe(df_raw, cfg)

    # Save cleaned data
    cleaned_path = root / cfg.cleaned_parquet_path
    save_parquet(df_clean, cleaned_path)

    # Build and run pipeline
    pipeline = build_preprocessing_pipeline(recipe, cfg)
    df_features = pipeline.fit_transform(df_clean)

    # Save features
    features_path = root / cfg.get_features_path(recipe)
    save_parquet(df_features, features_path)

    # Generate quality report
    report_path = root / cfg.reports_dir / "data_quality.json"
    report = generate_data_quality_report(df_features, report_path)

    # Print summary
    _print_summary(df_features, recipe, report)


def _print_summary(df, recipe: str, report: dict) -> None:
    """Print summary table."""
    table = Table(title="Preprocessing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Recipe", recipe)
    table.add_row("Rows", str(len(df)))
    table.add_row("Features", str(len(df.columns)))
    table.add_row("Date Range", f"{report['date_range']['start']} → {report['date_range']['end']}")
    table.add_row("NaN Total", str(sum(report["nan_counts"].values())))

    console.print(table)
    console.print("\n[bold]Columns:[/bold]")
    console.print(", ".join(df.columns.tolist()))


@app.command()
def list_recipes() -> None:
    """List available preprocessing recipes."""
    console.print("[bold]Available Recipes:[/bold]")
    recipes_info = {
        "baseline_z": "YC_SLOPE + rolling z-scores on levels only",
        "z_plus_momentum": "baseline_z + Δ1M/Δ6M + SPX drawdown",
        "changes_only": "Only diffs/momentum, no level z-scores",
        "levels_only": "Only level z-scores, no diffs",
    }
    for name, desc in recipes_info.items():
        console.print(f"  [green]{name}[/green]: {desc}")


@app.command()
def validate(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to raw data file"),
    ] = None,
) -> None:
    """Validate raw data file without processing."""
    root = get_project_root()
    cfg = PreprocessConfig()

    if input_path is None:
        input_path = root / cfg.raw_data_path

    console.print(f"[bold]Validating: {input_path}[/bold]")

    df_raw = load_raw_data(input_path, cfg)
    validate_dataframe(df_raw, cfg)
    warnings = check_monthly_frequency(df_raw)

    if warnings:
        for w in warnings:
            console.print(f"[yellow]Warning: {w}[/yellow]")
    else:
        console.print("[green]Validation passed![/green]")

    console.print(f"Rows: {len(df_raw)}")
    console.print(f"Columns: {df_raw.columns.tolist()}")


if __name__ == "__main__":
    app()


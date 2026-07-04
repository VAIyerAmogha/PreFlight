import os
import json
import joblib
from pathlib import Path
import typer
import pandas as pd
import preflight
from preflight.types import PrepResult

app = typer.Typer()

@app.callback()
def main():
    pass

def write_outputs(
    result: PrepResult,
    input_path: str,
    output_dir: str,
) -> dict[str, str]:
    written = {}
    name = Path(input_path).stem
    out_dir = Path(output_dir)
    
    csv_path = out_dir / f"{name}_prepared.csv"
    pipeline_path = out_dir / f"{name}_pipeline.joblib"
    report_path = out_dir / f"{name}_report.json"
    
    try:
        result.df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)
        
        if result.pipeline is not None:
            joblib.dump(result.pipeline, pipeline_path)
            written["pipeline"] = str(pipeline_path)
            
        if result.report is not None:
            with open(report_path, "w") as f:
                json.dump(result.report.to_dict(), f, indent=2)
            written["report"] = str(report_path)
            
    except OSError as e:
        typer.echo(f"Error writing output file: {e}", err=True)
        raise typer.Exit(code=1)
        
    return written

@app.command()
def prepare(
    input_path: str = typer.Argument(..., help="Path to input CSV file"),
    target: str = typer.Option(..., "--target", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="regression|classification"),
    model_hint: str = typer.Option("tree", "--model-hint", help="tree|linear"),
    drop_threshold: float = typer.Option(0.6, "--drop-threshold"),
    outlier_method: str = typer.Option("iqr", "--outlier-method"),
    cardinality_threshold: int = typer.Option(20, "--cardinality-threshold"),
    output_dir: str = typer.Option(".", "--output-dir", help="Directory for output files"),
    verbose: bool = typer.Option(False, "--verbose", help="Show full decision log inline"),
) -> None:
    if not (0 <= drop_threshold <= 1):
        typer.echo(f"Error: drop-threshold must be between 0 and 1, got {drop_threshold}", err=True)
        raise typer.Exit(code=1)
    
    if cardinality_threshold <= 0:
        typer.echo(f"Error: cardinality-threshold must be > 0, got {cardinality_threshold}", err=True)
        raise typer.Exit(code=1)
        
    if not os.path.exists(input_path) or not input_path.lower().endswith(".csv"):
        typer.echo("Error: input_path must exist and be a .csv file", err=True)
        raise typer.Exit(code=1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
    except pd.errors.ParserError as e:
        typer.echo(f"Error: could not parse CSV: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading CSV: {e}", err=True)
        raise typer.Exit(code=1)
        
    if target in df.columns and df[target].isnull().all():
        typer.echo(f"Error: Target column '{target}' is entirely null.", err=True)
        raise typer.Exit(code=1)
        
    try:
        result = preflight.prepare(
            df=df,
            target=target,
            task=task,
            model_hint=model_hint,
            drop_threshold=drop_threshold,
            outlier_method=outlier_method,
            cardinality_threshold=cardinality_threshold
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
        
    written_paths = write_outputs(result, input_path, output_dir)
    
    if verbose and result.report is not None:
        result.report.show()
    
    typer.echo(f"Success! Resulting shape: {result.df.shape}")
    for key, path in written_paths.items():
        typer.echo(f"Wrote {key}: {path}")

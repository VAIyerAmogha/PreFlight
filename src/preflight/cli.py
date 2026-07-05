import os
import json
import joblib
from pathlib import Path
import typer
import pandas as pd
import preflight
from preflight.types import PrepResult, FeatureConfig, SemanticType, PRESETS
from typing import Optional

app = typer.Typer(rich_markup_mode=None)

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
    preset: Optional[str] = typer.Option(None, "--preset", help="Named configuration preset"),
    drop_threshold: Optional[float] = typer.Option(None, "--drop-threshold"),
    outlier_method: Optional[str] = typer.Option(None, "--outlier-method"),
    cardinality_threshold: Optional[int] = typer.Option(None, "--cardinality-threshold"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory for output files"),
    verbose: bool = typer.Option(False, "--verbose", help="Show full decision log inline"),
    interactions: Optional[bool] = typer.Option(None, "--interactions/--no-interactions", help="Enable feature interactions"),
    interaction_top_k: Optional[int] = typer.Option(None, "--interaction-top-k", help="Top K features to interact"),
    interaction_types: Optional[str] = typer.Option(None, "--interaction-types", help="Comma-separated interaction types"),
    datetime_cyclical: Optional[bool] = typer.Option(None, "--datetime-cyclical/--no-datetime-cyclical", help="Enable cyclical datetime features"),
    datetime_deltas: Optional[bool] = typer.Option(None, "--datetime-deltas/--no-datetime-deltas", help="Enable datetime delta features"),
    datetime_reference_col: Optional[str] = typer.Option(None, "--datetime-reference-col", help="Reference datetime column"),
    clustering: Optional[bool] = typer.Option(None, "--clustering/--no-clustering", help="Enable KMeans clustering features"),
    cluster_k: Optional[str] = typer.Option(None, "--cluster-k", help="Number of clusters or 'auto'"),
    cluster_features: Optional[str] = typer.Option(None, "--cluster-features", help="Comma-separated cluster features"),
    text_features: Optional[bool] = typer.Option(None, "--text-features/--no-text-features", help="Enable text feature engineering"),
    text_tfidf: Optional[bool] = typer.Option(None, "--text-tfidf/--no-text-tfidf", help="Enable TF-IDF vectorization for text"),
    text_tfidf_top_k: Optional[int] = typer.Option(None, "--text-tfidf-top-k", help="Top K features for TF-IDF"),
    column_type: Optional[list[str]] = typer.Option(None, "--column-type", help="Override semantic type for a column (format colname:TYPE)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run full decision logic without transforming data or fitting a pipeline"),
    save_pdf: Optional[str] = typer.Option(None, "--save-pdf", help="Save graphical PDF report to this path"),
) -> None:
    if preset is not None and preset not in PRESETS:
        typer.echo(f"Error: Invalid preset '{preset}'. Valid presets are: {list(PRESETS.keys())}", err=True)
        raise typer.Exit(code=1)

    if drop_threshold is not None and not (0 <= drop_threshold <= 1):
        typer.echo(f"Error: drop-threshold must be between 0 and 1, got {drop_threshold}", err=True)
        raise typer.Exit(code=1)
    
    if cardinality_threshold is not None and cardinality_threshold <= 0:
        typer.echo(f"Error: cardinality-threshold must be > 0, got {cardinality_threshold}", err=True)
        raise typer.Exit(code=1)
        
    if not os.path.exists(input_path) or not input_path.lower().endswith(".csv"):
        typer.echo("Error: input_path must exist and be a .csv file", err=True)
        raise typer.Exit(code=1)
        
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."
        
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
        
    explicit_feature_args = {
        "interactions": interactions,
        "interaction_top_k": interaction_top_k,
        "interaction_types": interaction_types,
        "datetime_cyclical": datetime_cyclical,
        "datetime_deltas": datetime_deltas,
        "datetime_reference_col": datetime_reference_col,
        "clustering": clustering,
        "cluster_k": cluster_k,
        "cluster_features": cluster_features,
        "text_features": text_features,
        "text_tfidf": text_tfidf,
        "text_tfidf_top_k": text_tfidf_top_k,
    }
    provided_feature_args = {k: v for k, v in explicit_feature_args.items() if v is not None}
    
    use_feature_config = len(provided_feature_args) > 0
    
    feature_config = None
    if use_feature_config:
        base_fc = None
        if preset is not None and preset in PRESETS:
            base_fc = PRESETS[preset].get("feature_config")
            
        fc_kwargs = {}
        if base_fc is not None:
            from dataclasses import asdict
            fc_kwargs = asdict(base_fc)
            
        if "cluster_k" in provided_feature_args:
            val = provided_feature_args["cluster_k"]
            if val != "auto":
                try:
                    provided_feature_args["cluster_k"] = int(val)
                except ValueError:
                    typer.echo(f"Error: cluster-k must be 'auto' or an integer, got '{val}'", err=True)
                    raise typer.Exit(code=1)
        if "cluster_features" in provided_feature_args:
            val = provided_feature_args["cluster_features"]
            if val != "numeric_only":
                provided_feature_args["cluster_features"] = [x.strip() for x in val.split(",")]
        if "interaction_types" in provided_feature_args:
            provided_feature_args["interaction_types"] = [x.strip() for x in provided_feature_args["interaction_types"].split(",")]
            
        fc_kwargs.update(provided_feature_args)
        
        try:
            feature_config = FeatureConfig(**fc_kwargs)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
            
    parsed_column_types = None
    if column_type:
        parsed_column_types = {}
        for ct in column_type:
            parts = ct.split(":", 1)
            if len(parts) != 2:
                typer.echo(f"Error: Invalid format for --column-type '{ct}'. Expected format is colname:TYPE", err=True)
                raise typer.Exit(code=1)
            cname, ctype_str = parts
            try:
                ctype_enum = SemanticType[ctype_str]
            except KeyError:
                typer.echo(f"Error: '{ctype_str}' is not a valid SemanticType", err=True)
                raise typer.Exit(code=1)
            parsed_column_types[cname] = ctype_enum
            
    kwargs = {
        "df": df,
        "target": target,
        "task": task,
        "model_hint": model_hint,
    }
    
    if preset is not None:
        kwargs["preset"] = preset
    if drop_threshold is not None:
        kwargs["drop_threshold"] = drop_threshold
    if outlier_method is not None:
        kwargs["outlier_method"] = outlier_method
    if cardinality_threshold is not None:
        kwargs["cardinality_threshold"] = cardinality_threshold
    if use_feature_config:
        kwargs["feature_config"] = feature_config
    if parsed_column_types is not None:
        kwargs["column_types"] = parsed_column_types
    if dry_run:
        kwargs["dry_run"] = True

    try:
        result = preflight.prepare(**kwargs)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
        
    written_paths = write_outputs(result, input_path, output_dir)
    
    if save_pdf and result.report is not None:
        result.report.save_pdf(save_pdf)
        written_paths["pdf"] = save_pdf
    
    if verbose and result.report is not None:
        result.report.show()
    
    typer.echo(f"Success! Resulting shape: {result.df.shape}")
    for key, path in written_paths.items():
        typer.echo(f"Wrote {key}: {path}")

@app.command("compare")
def compare_cli(
    csv_a: str = typer.Argument(..., help="Path to first CSV"),
    csv_b: str = typer.Argument(..., help="Path to second CSV"),
    target_a: str = typer.Option(..., "--target-a", help="Target column for first CSV"),
    target_b: str = typer.Option(..., "--target-b", help="Target column for second CSV"),
    output_pdf: str = typer.Option(..., "--output-pdf", help="Path to save output PDF report"),
    task_a: str = typer.Option("classification", "--task-a", help="Task for first CSV"),
    task_b: str = typer.Option("classification", "--task-b", help="Task for second CSV"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Preset to use for both runs"),
):
    typer.echo("Running prepare on first dataset...")
    try:
        df_a = pd.read_csv(csv_a)
        res_a = preflight.prepare(df_a, target=target_a, task=task_a, preset=preset)
    except Exception as e:
        typer.echo(f"Error preparing first dataset: {e}", err=True)
        raise typer.Exit(code=1)
        
    typer.echo("Running prepare on second dataset...")
    try:
        df_b = pd.read_csv(csv_b)
        res_b = preflight.prepare(df_b, target=target_b, task=task_b, preset=preset)
    except Exception as e:
        typer.echo(f"Error preparing second dataset: {e}", err=True)
        raise typer.Exit(code=1)
        
    typer.echo("Generating comparison report...")
    try:
        from preflight.compare_report import save_compare_pdf
        save_compare_pdf(res_a, res_b, output_pdf)
        typer.echo(f"Successfully wrote comparison PDF to: {output_pdf}")
    except Exception as e:
        typer.echo(f"Error generating comparison PDF: {e}", err=True)
        raise typer.Exit(code=1)


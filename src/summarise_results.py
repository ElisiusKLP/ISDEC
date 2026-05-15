from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

results_dir = Path("results/classification")
summary_dir = Path("results/summary")
summary_dir.mkdir(parents=True, exist_ok=True)

def load_all_subjects_into_dataframe(results_dir: Path):
    """Traverse the results directory and print summary of results."""
    all_results = []
    all_paths = list(results_dir.glob("**/*.joblib"))
    if not all_paths:
        raise ValueError(f"No result files found in {results_dir.resolve()}")
    for result_file in all_paths:
        result = joblib.load(result_file)

        y_true = result.get("y_true")
        y_pred = result.get("y_pred")
        score = result.get("score", "Na")
        delta_over_chance = result.get("delta_over_chance", "Na")
        confusion_matrix = result.get("confusion_matrix", "Na")
        model_info = result.get("model_info", {})
        model_name = model_info.get("model_name", "Na")
        feature_type = model_info.get("feature_type", "Na")
        scale = model_info.get("scale", "Na")
        config = model_info.get("config", None)
        subject_id = result.get("subject_id", "Na")

        all_results.append({
            "subject_id": subject_id,
            "model_name": model_name,
            "feature_type": feature_type,
            "scale": scale,
            "score": score,
            "delta_over_chance": delta_over_chance,
            "confusion_matrix": confusion_matrix,
            "config": config,
        })

    df = pd.DataFrame(all_results)
    # save as csv
    csv_path = summary_dir / "classification_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path.resolve()}")

def plot_all_models_average_score(summary_df: pd.DataFrame):
    "Plot scatter plots with models on x-axis and scores on y-axis, with different colors for features, scale == True"
    def _is_scaled(value):
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ("scale", "scaled"):
                return True
            if value_lower in ("no_scale", "noscale"):
                return False
            return value_lower in ("true", "t", "1", "yes", "y")
        return bool(value)

    plot_df = summary_df.copy()
    plot_df["is_scaled"] = plot_df["scale"].map(_is_scaled)

    feature_types = plot_df["feature_type"].unique()
    color_map = {feature: px.colors.qualitative.Plotly[i % len(px.colors
    .qualitative.Plotly)] for i, feature in enumerate(feature_types)}

    fig = px.scatter(
        plot_df,
        x="model_name",
        y="score",
        color="feature_type",
        symbol="is_scaled",
        hover_data={"feature_type": True, "scale": True, "model_name": False, "score": ":.4f"},
        color_discrete_map=color_map,
        symbol_map={True: "circle", False: "square"},
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=0.5, color="rgba(0, 0, 0, 0.35)")))
    fig.update_layout(
        title="Model Performance by Feature Type",
        xaxis_title="Model Name",
        yaxis_title="Score",
        legend_title="Model Name",
        width=1200,
        height=600,
    )
    fig.update_xaxes(tickangle=45)

    plot_path = summary_dir / "model_performance_by_feature_type.html"
    fig.write_html(plot_path)
    print(f"Saved model performance plot to {plot_path.resolve()}")

def plot_within_model_comparison(summary_df: pd.DataFrame):
    """Plot grid-based hyperparameter results with config information.
    Filters to only results from /grid/ subfolders (where config was used).
    Groups by model, feature, scale, and config to compare hyperparameter combinations.
    """
    # Filter to only grid results (non-null config)
    grid_df = summary_df[summary_df["config"].notna()].copy()
    
    if grid_df.empty:
        print("No grid-based results found (no config parameters used).")
        return
    
    def _is_scaled(value):
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ("scale", "scaled"):
                return True
            if value_lower in ("no_scale", "noscale"):
                return False
            return value_lower in ("true", "t", "1", "yes", "y")
        return bool(value)
    
    grid_df["is_scaled"] = grid_df["scale"].map(_is_scaled)
    
    # Convert config dict to string representation for display
    grid_df["config_str"] = grid_df["config"].apply(
        lambda x: str(x) if isinstance(x, dict) else str(x)
    )
    
    # Group by model, feature, scale, config and calculate mean score
    grouped_df = grid_df.groupby(
        ["model_name", "feature_type", "scale", "config_str"],
        as_index=False
    )["score"].mean()
    grouped_df["is_scaled"] = grouped_df["scale"].map(_is_scaled)
    
    feature_types = grouped_df["feature_type"].unique()
    color_map = {
        feature: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, feature in enumerate(feature_types)
    }
    
    fig = px.scatter(
        grouped_df,
        x="model_name",
        y="score",
        color="feature_type",
        symbol="is_scaled",
        hover_data={
            "feature_type": True,
            "scale": True,
            "config_str": True,
            "model_name": False,
            "score": ":.4f",
        },
        color_discrete_map=color_map,
        symbol_map={True: "circle", False: "square"},
    )
    
    fig.update_traces(marker=dict(size=12, line=dict(width=0.5, color="rgba(0, 0, 0, 0.35)")))
    fig.update_layout(
        title="Model Performance by Feature Type & Hyperparameter Config",
        xaxis_title="Model Name",
        yaxis_title="Score",
        legend_title="Feature Type",
        width=1200,
        height=600,
    )
    fig.update_xaxes(tickangle=45)
    
    plot_path = summary_dir / "model_performance_by_config_grid.html"
    fig.write_html(plot_path)
    print(f"Saved grid hyperparameter plot to {plot_path.resolve()}")
    
    # Also save a summary CSV of grid results
    grid_csv_path = summary_dir / "classification_summary_grid.csv"
    grouped_df.to_csv(grid_csv_path, index=False)
    print(f"Saved grid results summary to {grid_csv_path.resolve()}")

def summarise_results():

    load_all_subjects_into_dataframe(results_dir)

    summary_df = pd.read_csv(summary_dir / "classification_summary.csv")

    # Plot grid-based results (if any exist)
    plot_within_model_comparison(summary_df)

    # aggregate summary by model, feature_type and scale, and calculate mean score across subjects
    summary_df_aggregated = summary_df.groupby(["model_name", "feature_type", "scale"], as_index=False)["score"].mean()
    summary_df_aggregated.head(10)
    # save the summary df as csv
    summary_csv_path = summary_dir / "classification_summary_aggregated.csv"
    summary_df_aggregated.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated summary CSV to {summary_csv_path.resolve()}")

    plot_all_models_average_score(summary_df_aggregated)




if __name__ == "__main__":
    summarise_results()
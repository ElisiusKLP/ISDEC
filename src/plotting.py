from pyexpat import model
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from jax.experimental.hijax import HiPrimitive
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_confusion_matrices_grid(img_dir: Path, set_tag: str):
    """Plot confusion matrices in a grid and save as PDF"""
    imgs = sorted(img_dir.glob("*.png"))
    imgs = [img for img in imgs if set_tag in img.stem]
    if not imgs:
        print(f"No images found for set '{set_tag}' in {img_dir.resolve()}")
        raise ValueError(f"No images found for set '{set_tag}'")

    n_cols = 3
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()

    for ax, img_path in zip(axes, imgs[:n_rows*n_cols]):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=9)
        ax.axis("off")

    # hide any extra axes if fewer than grid slots
    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    out_pdf = img_dir / f"all_confusion_matrices_grid_{set_tag}.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf.resolve(), bbox_inches="tight")
    print(f"Saved grid PDF to {out_pdf.resolve()}")


# ============================================================================
# Theme Configuration - Minimal style with orange/green color palette
# ============================================================================

# Diverse categorical palette, ordered so nearby categories cycle across groups.
THEME_COLORS = {
    "navy_deep": "#00202e",
    "navy": "#003f5c",
    "blue": "#2c4875",
    "purple": "#8a508f",
    "magenta": "#bc5090",
    "coral": "#ff6361",
    "orange": "#ff8531",
    "amber": "#ffa600",
    "sand": "#ffd380",
}

# Interleave the palette by group so the first categories are visually distinct.
CATEGORICAL_PALETTE = [
    THEME_COLORS["blue"],
    THEME_COLORS["coral"],
    THEME_COLORS["amber"],
    THEME_COLORS["navy_deep"],
    THEME_COLORS["orange"],
    THEME_COLORS["purple"],
    THEME_COLORS["navy"],
    THEME_COLORS["sand"],
    THEME_COLORS["magenta"]
]

def get_color_map(items):
    """Generate a color map for categorical items using orange/green palette."""
    return {
        item: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        for i, item in enumerate(items)
    }

def apply_theme(fig, title="", xaxis_title="", yaxis_title="", legend_title="", width=1200, height=600,
        add_hline=True
        ):
    """Apply minimal theme with consistent styling across all plots."""
    fig.update_layout(
        template="plotly_white",  # minimal template
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#333333"),
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(
            title=legend_title,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#CCCCCC",
            borderwidth=0,
        ),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman", size=12, color="#333333"),
        hovermode="closest",
    )
    
    # Minimal grid styling
    fig.update_xaxes(
        showgrid=False,
        gridwidth=1,
        gridcolor="#EEEEEE",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#EEEEEE",
        zeroline=False,
    )
    
    # Enhance marker styling
    fig.update_traces(
        marker=dict(
            size=10,
            line=dict(width=0.5, color="rgba(0, 0, 0, 0.3)"),
            opacity=0.8,
        ),
        selector=dict(mode="markers"),
    )

    if add_hline:
        fig.add_hline(
            y=0.2,
            line_dash="dash",
            line_color="gray",
            line_width=1,
        )
    
    return fig

###
# PLOTLY PLOTS
###
# Model name to label mapping
MODEL_LABEL_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svc": "SVM"
}

# Feature type to display label mapping
FEATURE_LABEL_MAP = {
    "stack": "Stacked",
    "mean": "Mean",
    "mean_mi": "Mean (MI)",
    "bandpower_mean": "Bandpower Mean",
    "bandpower_mean_mi": "Bandpower Mean (MI)",
    "dwt_stats": "DWT Stats",
    "dwt_stats_mi": "DWT Stats (MI)"
}

def remap_labels(df):
    # Store originals
    new_df = df.copy()
    original_model_names = new_df["model_name"].copy()
    original_feature_types = new_df["feature_type"].copy()

    # Apply mapping
    new_df["model_name"] = new_df["model_name"].map(MODEL_LABEL_MAP)
    new_df["feature_type"] = new_df["feature_type"].map(FEATURE_LABEL_MAP)

    # Detect unmapped model names
    missing_models = original_model_names[new_df["model_name"].isna()].unique()
    if len(missing_models) > 0:
        print("Unmapped model_name values:")
        for val in missing_models:
            print(f"  - {repr(val)}")

    # Detect unmapped feature types
    missing_features = original_feature_types[new_df["feature_type"].isna()].unique()
    if len(missing_features) > 0:
        print("Unmapped feature_type values:")
        for val in missing_features:
            print(f"  - {repr(val)}")

    return new_df

def plot_all_models_average_score(summary_df: pd.DataFrame,
output_dir: Path):
    """Plot scatter plots with models on x-axis and scores on y-axis, colored by feature type."""
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

    feature_types = sorted(plot_df["feature_type"].unique())
    color_map = get_color_map(feature_types)

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

    fig = apply_theme(
        fig,
        title="Model Performance by Feature Extraction",
        xaxis_title="Model Name",
        yaxis_title="Mean Accuracy",
        legend_title="Feature Extraction",
    )
    fig.update_xaxes(tickangle=45)

    plot_path = output_dir / "model_performance_by_feature_type.html"
    fig.write_html(plot_path)
    print(f"Saved model performance plot to {plot_path.resolve()}")


def plot_within_model_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Plot grid-based hyperparameter results with config information."""
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
    grid_df["config_str"] = grid_df["config"].apply(
        lambda x: str(x) if isinstance(x, dict) else str(x)
    )
    
    grouped_df = grid_df.groupby(
        ["model_name", "feature_type", "scale", "config_str"],
        as_index=False
    )["score"].mean()
    grouped_df["is_scaled"] = grouped_df["scale"].map(_is_scaled)
    
    feature_types = sorted(grouped_df["feature_type"].unique())
    color_map = get_color_map(feature_types)
    
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
    
    fig = apply_theme(
        fig,
        title="Model Performance by Feature Type & Hyperparameter Config",
        xaxis_title="Model Name",
        yaxis_title="Score",
        legend_title="Feature Type",
    )
    fig.update_xaxes(tickangle=45)
    
    plot_path = output_dir / "model_performance_by_config_grid.html"
    fig.write_html(plot_path)
    print(f"Saved grid hyperparameter plot to {plot_path.resolve()}")
    
    grid_csv_path = output_dir / "classification_summary_grid.csv"
    grouped_df.to_csv(grid_csv_path, index=False)
    print(f"Saved grid results summary to {grid_csv_path.resolve()}")


def plot_top_fit_per_feature(summary_df: pd.DataFrame, output_dir: Path):
    """Plot top fit for each model within each feature type."""
    top_fit_df = (
        summary_df
        .sort_values("score", ascending=False)
        .groupby(["feature_type", "model_name"], as_index=False)
        .first()
    )
    top_fit_df = remap_labels(top_fit_df)

    feature_types = list(FEATURE_LABEL_MAP.values())
    models = list(MODEL_LABEL_MAP.values())

    group_spacing = 2.0
    model_spacing = 0.25

    feature_centers = {
        ft: i * group_spacing
        for i, ft in enumerate(feature_types)
    }

    n_models = len(models)
    model_offsets = {
        model: (i - (n_models - 1) / 2) * model_spacing
        for i, model in enumerate(models)
    }

    top_fit_df["x_pos"] = top_fit_df.apply(
        lambda row:
            feature_centers[row["feature_type"]]
            + model_offsets[row["model_name"]],
        axis=1,
    )

    model_color_map = get_color_map(models)

    fig = px.scatter(
        top_fit_df,
        x="x_pos",
        y="score",
        color="model_name",
        error_y="score_std",
        hover_data={
            "feature_type": True,
            "model_name": True,
            "score": ":.4f",
            "score_std": ":.4f",
            "x_pos": False,
            "config": True
        },
        color_discrete_map=model_color_map,
    )

    fig = apply_theme(
        fig,
        title="Top Model Performance by Feature Extraction",
        xaxis_title="Feature Extraction",
        yaxis_title="Mean Accuracy",
        legend_title="Model"
    )
    
    fig.update_layout(
        xaxis=dict(
            tickvals=[feature_centers[ft] for ft in feature_types],
            ticktext=feature_types,
            tickangle=0
        ),
        yaxis=dict(
        range=[0, 1]  # Replace y_min and y_max with your desired values
        )
    )

    plot_path = output_dir / "top_model_performance_by_feature_type.html"
    fig.write_html(plot_path)
    print(f"Saved plot to {plot_path.resolve()}")


def plot_mean_sd_plot(summary_df_aggregated: pd.DataFrame, output_dir: Path):
    """Plot scatter with mean score on y-axis and standard deviation on x-axis."""
    def _is_scaled(value):
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ("scale", "scaled"):
                return True
            if value_lower in ("no_scale", "noscale"):
                return False
            return value_lower in ("true", "t", "1", "yes", "y")
        return bool(value)

    plot_df = summary_df_aggregated.copy()
    plot_df["is_scaled"] = plot_df["scale"].map(_is_scaled)
    plot_df = plot_df[plot_df["is_scaled"] == True]

    plot_df = remap_labels(plot_df)

    model_names = list(MODEL_LABEL_MAP.values())
    model_color_map = get_color_map(model_names)

    fig = px.scatter(
        plot_df,
        x="score_std",
        y="score",
        color="model_name",
        hover_data={"feature_type": True, "scale": True, "model_name": True, "score": ":.4f", "score_std": ":.4f", "config": True},
        color_discrete_map=model_color_map,
    )

    fig = apply_theme(
        fig,
        title="Model Performance: Mean vs Standard Deviation",
        xaxis_title="Accuracy SD",
        yaxis_title="Mean Accuracy",
        legend_title="Model",
    )

    plot_path = output_dir / "model_performance_mean_sd_by_feature_type.html"
    fig.write_html(plot_path)
    print(f"Saved mean vs sd performance plot to {plot_path.resolve()}")

def plot_violin_per_feature(summary_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot of scores for each feature type."""
    plot_df = remap_labels(summary_df)

    feature_types = list(FEATURE_LABEL_MAP.values())
    color_map = get_color_map(feature_types)

    fig = px.violin(
        plot_df,
        x="feature_type",
        y="score",
        color="feature_type",
        hover_data={"model_name": True, "scale": True, "score": ":.4f"},
        color_discrete_map=color_map,
    )

    fig = apply_theme(
        fig,
        title="Accuracy Distribution by Feature Extraction",
        xaxis_title="Feature Extraction",
        yaxis_title="Accuracy",
        legend_title="Feature Extraction",
    )
    fig.update_xaxes(tickangle=0)

    plot_path = output_dir / "score_distribution_by_feature_type.html"
    fig.write_html(plot_path)
    print(f"Saved violin plot to {plot_path.resolve()}")

def plot_violin_per_model(summary_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot of scores for each model."""
    print(summary_df["model_name"].unique())
    print(summary_df["feature_type"].unique())
    plot_df = remap_labels(summary_df)
    model_names = list(MODEL_LABEL_MAP.values())
    print(f"model_names: {model_names}")
    print(plot_df.head(2))
    color_map = get_color_map(model_names)

    fig = px.violin(
        plot_df,
        x="model_name",
        y="score",
        color="model_name",
        hover_data={"feature_type": True, "scale": True, "score": ":.4f"},
        color_discrete_map=color_map,
    )

    fig = apply_theme(
        fig,
        title="Accuracy Distribution by Model",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        legend_title="Model",
    )
    fig.update_xaxes(tickangle=0)

    plot_path = output_dir / "score_distribution_by_model.html"
    fig.write_html(plot_path)
    print(f"Saved violin plot to {plot_path.resolve()}")

def plot_violin_per_model_and_feature(summary_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot of scores for each model and feature type."""
    fig = px.violin(
        summary_df,
        x="model_name",
        y="score",
        color="feature_type",
        hover_data={"scale": True, "score": ":.4f"},
        color_discrete_map=get_color_map(sorted(summary_df["feature_type"].unique())),
    )

    fig = apply_theme(
        fig,
        title="Score Distribution by Model and Feature Type",
        xaxis_title="Model Name",
        yaxis_title="Score",
        legend_title="Feature Type",
    )
    fig.update_xaxes(tickangle=45)

    plot_path = output_dir / "score_distribution_by_model_and_feature.html"
    fig.write_html(plot_path)
    print(f"Saved violin plot to {plot_path.resolve()}")

def plot_violin_per_model_with_feature_scatter(summary_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot of scores for each model with scatter points colored by feature type."""
    model_names = sorted(summary_df["model_name"].unique())
    feature_types = sorted(summary_df["feature_type"].unique())
    feature_color_map = get_color_map(feature_types)
    model_to_x = {model_name: i for i, model_name in enumerate(model_names)}
    violin_width = 0.8
    jitter_width = violin_width * 0.28
    rng = np.random.default_rng(0)

    fig = go.Figure()

    # Draw the violin shells in black so the feature colors read as the interior scatter.
    for model_name in model_names:
        model_df = summary_df[summary_df["model_name"] == model_name]
        fig.add_trace(
            go.Violin(
                x=[model_to_x[model_name]] * len(model_df),
                y=model_df["score"],
                name=model_name,
                legendgroup=model_name,
                showlegend=False,
                line=dict(color="black", width=1),
                fillcolor="rgba(0, 0, 0, 0.15)",
                opacity=1,
                points=False,
                hoverinfo="skip",
                scalemode="width",
                width=violin_width,
            )
        )

    # Overlay the scatter points for each feature type.
    for feature_type in feature_types:
        feature_df = summary_df[summary_df["feature_type"] == feature_type]
        feature_x = feature_df["model_name"].map(model_to_x).to_numpy(dtype=float)
        jitter_offsets = rng.uniform(-jitter_width, jitter_width, size=len(feature_df))
        fig.add_trace(
            go.Scatter(
                x=feature_x + jitter_offsets,
                y=feature_df["score"],
                mode="markers",
                name=feature_type,
                marker=dict(
                    color=feature_color_map[feature_type],
                    size=8,
                    line=dict(width=0.4, color="rgba(0, 0, 0, 0.35)"),
                    opacity=0.85,
                ),
                customdata=np.stack(
                    [feature_df["model_name"].astype(str), feature_df["scale"].astype(str), feature_df["score"].astype(float)],
                    axis=-1,
                ),
                hovertemplate=(
                    "Model: %{customdata[0]}<br>"
                    "Feature: " + feature_type + "<br>"
                    "Scale: %{customdata[1]}<br>"
                    "Score: %{customdata[3]:.4f}<extra></extra>"
                ),
            )
        )

    # Compute mean and std for each (model, feature_type) and plot as larger markers with error bars
    agg = (
        summary_df
        .groupby(["model_name", "feature_type"], as_index=False)["score"]
        .agg(["mean", "std"])  # mean and std per group
        .reset_index()
    )

    for _, row in agg.iterrows():
        mx = model_to_x.get(row["model_name"])
        if mx is None:
            continue
        ft = row["feature_type"]
        mean_val = row[("mean")]
        std_val = row[("std")]
        fig.add_trace(
            go.Scatter(
                x=[mx],
                y=[mean_val],
                mode="markers",
                name=ft,
                showlegend=False,
                marker=dict(
                    color=feature_color_map.get(ft, "black"),
                    size=12,
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                ),
                error_y=dict(
                    type="data",
                    array=[std_val if not np.isnan(std_val) else 0.0],
                    visible=True,
                    thickness=1.5,
                    width=6,
                ),
                hovertemplate=(
                    "Model: %s<br>Feature: %s<br>Mean: %%{y:.4f}<br>Std: %.4f<extra></extra>" % (row["model_name"], ft, std_val)
                ),
            )
        )

    fig = apply_theme(
        fig,
        title="Score Distribution by Model with Feature Type Scatter",
        xaxis_title="Model Name",
        yaxis_title="Score",
        legend_title="Feature Type",
    )
    fig.update_xaxes(tickangle=45)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(model_to_x.values()),
        ticktext=model_names,
        range=[-0.6, len(model_names) - 0.4],
    )
    fig.update_traces(orientation="v")

    plot_path = output_dir / "score_distribution_by_model_with_feature_scatter.html"
    fig.write_html(plot_path)
    print(f"Saved violin plot with scatter to {plot_path.resolve()}")

def plot_violin_per_model_with_feature_error_bars(summary_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot of scores for each model with error bars for each feature type."""
    model_names = sorted(summary_df["model_name"].unique())
    feature_types = sorted(summary_df["feature_type"].unique())
    feature_color_map = get_color_map(feature_types)
    model_to_x = {model_name: i for i, model_name in enumerate(model_names)}
    violin_width = 0.8

    fig = go.Figure()

    # Draw the violin shells in black so the feature colors read as the interior scatter.
    for model_name in model_names:
        model_df = summary_df[summary_df["model_name"] == model_name]
        fig.add_trace(
            go.Violin(
                x=[model_to_x[model_name]] * len(model_df),
                y=model_df["score"],
                name=model_name,
                legendgroup=model_name,
                showlegend=False,
                line=dict(color="black", width=1),
                fillcolor="rgba(0, 0, 0, 0.15)",
                opacity=1,
                points=False,
                hoverinfo="skip",
                scalemode="width",
                width=violin_width,
            )
        )

    # Overlay the error bars for each feature type.
    for feature_type in feature_types:
        feature_df = summary_df[summary_df["feature_type"] == feature_type]
        error_y = dict(
            type="data",
            array=feature_df["score_std"],
            visible=True,
            color=feature_color_map[feature_type],
            thickness=2,
            width=4,
        )
        fig.add_trace(
            go.Scatter(
                x=feature_df["model_name"].map(model_to_x).to_numpy(dtype=float),
                y=feature_df["score"],
                mode="markers",
                name=feature_type,
                marker=dict(
                    color=feature_color_map[feature_type],
                    size=8,
                    line=dict(width=0.4, color="rgba(0, 0, 0, 0.35)"),
                    opacity=0.85,
                ),
                error_y=error_y,
                customdata=np.stack(
                    [feature_df["model_name"].astype(str), feature_df["scale"].astype(str), feature_df["score"].astype(float)],
                    axis=-1,
                ),
                hovertemplate=(
                    "Model: %{customdata[0]}<br>"
                    "Feature: " + feature_type + "<br>"
                    "Scale: %{customdata[1]}<br>"
                    "Score: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

        fig = apply_theme(
            fig,
            title="Score Distribution by Model with Feature Type Error Bars",
            xaxis_title="Model Name",
            yaxis_title="Score",
            legend_title="Feature Type",
        )
        fig.update_xaxes(tickangle=45)
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(model_to_x.values()),
            ticktext=model_names,
            range=[-0.6, len(model_names) - 0.4],
        )
        fig.update_traces(orientation="v")

    plot_path = output_dir / "score_distribution_by_model_with_feature_error_bars.html"
    fig.write_html(plot_path)
    print(f"Saved violin plot with error bars to {plot_path.resolve()}")

def plot_mean_accuracy_per_feature(summary_df: pd.DataFrame, output_dir: Path, model_name: str):
    """Plot boxplot chart of mean accuracy for each feature type."""

    # Filter the DataFrame for the specified model name
    model_df = summary_df[summary_df["model_name"] == model_name]
    model_df = remap_labels(model_df)

    model_color_map = get_color_map(FEATURE_LABEL_MAP.values())

    model_name_label = MODEL_LABEL_MAP[model_name]

    fig_type = "box"
    if fig_type == "box":
        fig = px.box(
            model_df,
            x="feature_type",
            y="score",
            color="feature_type",
            hover_data={"model_name": True, "scale": True, "score": ":.4f"},
            color_discrete_map=model_color_map,
        )
    elif fig_type == "bar":
        agg_df = (
            model_df
            .groupby("feature_type", as_index=False)["score"]
            .agg(["mean", "std"])
            .reset_index()
        )
        fig = px.bar(
            agg_df,
            x="feature_type",
            y="mean",
            error_y="std",
            color="feature_type",
            hover_data={"mean": ":.4f", "std": ":.4f"},
            color_discrete_map=model_color_map,
        )

    fig = apply_theme(
        fig,
        title=f"{model_name_label} Performance by Feature Extraction",
        xaxis_title="Feature Extraction",
        yaxis_title="Accuracy",
        legend_title="Model",
        width=900,
        height=600,
    )

    fig.update_yaxes(range=[0, .5])  # Set y-axis range to [0, 1] for accuracy

    plot_path = output_dir / f"top_model_performance_by_feature_type_model-{model_name}.html"
    fig.write_html(plot_path)
    print(f"Saved plot to {plot_path.resolve()}")
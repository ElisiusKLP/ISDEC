from sympy.physics.units import s
from pyexpat import model
from matplotlib.pylab import plot
from jax.experimental.hijax import HiPrimitive
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from plotting import (
    plot_all_models_average_score,
    plot_mean_sd_plot,
    plot_top_fit_per_feature, plot_mean_accuracy_per_feature_all_models,
    plot_within_model_comparison,
    plot_top_grid_confusion_matrix_rates,
    plot_summary_table,
    plot_violin_per_feature,
    plot_violin_per_model,
    plot_violin_per_model_and_feature,
    plot_violin_per_model_with_feature_scatter,
    plot_violin_per_model_with_feature_error_bars,
    plot_mean_accuracy_per_feature,
    plot_mean_kfold_accuracy_per_proportion,
    plot_mean_train_pred_time_per_proportion,
    plot_kfold_summary_subplots,
)

results_dir = Path("results/classification")
kfold_dir = Path("results/kfold")

# output summary dir
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

def create_aggregated_feature_summary(summary_df: pd.DataFrame):
    """Aggregate summary by model, feature_type and scale, and calculate mean score across subjects."""
    summary_df_aggregated = summary_df.groupby(
        ["model_name", "feature_type", "scale"],
        as_index=False
    )["score"].mean()

    # create a column for std of score across subjects
    summary_df_aggregated["score_std"] = summary_df.groupby(
        ["model_name", "feature_type", "scale"]
    )["score"].std().values

    return summary_df_aggregated

def create_aggregated_feature_config_summary(summary_df: pd.DataFrame):
    """Aggregate summary by model, feature_type, scale and config, and calculate mean score across subjects."""
    summary_df_aggregated = summary_df.groupby(
        ["model_name", "feature_type", "scale", "config"],
        as_index=False
    )["score"].mean()

    # create a column for std of score across subjects
    summary_df_aggregated["score_std"] = summary_df.groupby(
        ["model_name", "feature_type", "scale", "config"]
    )["score"].std().values

    return summary_df_aggregated

def load_all_kfold_results_into_dataframe(kfold_dir: Path):
    """Traverse the kfold results directory and print summary of results."""
    all_results = []
    all_paths = list(kfold_dir.glob("**/*.joblib"))
    if not all_paths:
        raise ValueError(f"No kfold result files found in {kfold_dir.resolve()}")
    for result_file in all_paths:
        result = joblib.load(result_file)

        if isinstance(result, list):
            kfold_scores = result
            model_name = "Na"
            feature_type = "Na"
            scale = "Na"
            config = None
            subject_id = "Na"
            proportion = "Na"
        else:
            kfold_scores = result.get("scores", [])
            model_name = result.get("model_name", "Na")
            feature_type = result.get("feature_type", "Na")
            scale = result.get("scale", "Na")
            config = result.get("config", None)
            subject_id = result.get("subject_id", "Na")
            proportion = result.get("proportion", "Na")

        for fold_result in kfold_scores:
            fold_number = fold_result.get("fold", "Na")
            accuracy = fold_result.get("accuracy", "Na")
            precision = fold_result.get("precision", "Na")
            recall = fold_result.get("recall", "Na")
            f1 = fold_result.get("f1", "Na")
            train_time = fold_result.get("train_time", "Na")
            pred_time = fold_result.get("pred_time", "Na")

            all_results.append({
                "subject_id": subject_id,
                "model_name": model_name,
                "feature_type": feature_type,
                "scale": scale,
                "config": config,
                "fold": fold_number,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "proportion": proportion,
                "train_time": train_time,
                "pred_time": pred_time
            })

    df = pd.DataFrame(all_results)
    # save as csv
    csv_path = summary_dir / "classification_kfold_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved kfold summary CSV to {csv_path.resolve()}")

def create_aggregated_kfold_summary(kfold_dataframe: pd.DataFrame):
    """Aggregate kfold results by model, feature_type and scale, and calculate mean score across folds."""
    kfold_summary = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"],
        as_index=False
    )["accuracy"].mean()

    # create a column for std of accuracy across folds
    kfold_summary["accuracy_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["accuracy"].std().values

    # summarise precision, recall and f1 across folds
    kfold_summary["precision_mean"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["precision"].mean().values
    kfold_summary["precision_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]    )["precision"].std().values
    kfold_summary["recall_mean"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]    )["recall"].mean().values
    kfold_summary["recall_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]    )["recall"].std().values
    kfold_summary["f1_mean"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]    )["f1"].mean().values
    kfold_summary["f1_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]    )["f1"].std().values


    # summarise train_time and pred_time across folds
    kfold_summary["train_time_mean"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["train_time"].mean().values
    kfold_summary["train_time_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["train_time"].std().values
    kfold_summary["pred_time_mean"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["pred_time"].mean().values
    kfold_summary["pred_time_std"] = kfold_dataframe.groupby(
        ["model_name", "feature_type", "proportion"]
    )["pred_time"].std().values
    

    return kfold_summary


def summarise_results():

    

    summary_df_path = summary_dir / "classification_summary.csv"
    if not summary_df_path.exists():
        load_all_subjects_into_dataframe(results_dir)
        summary_df = pd.read_csv(summary_df_path)
    else:
        print(f"Summary CSV already exists at {summary_df_path.resolve()}, loading it...")
        summary_df = pd.read_csv(summary_df_path)

    # Plot grid-based results (if any exist)
    plot_within_model_comparison(
        summary_df,
        output_dir=summary_dir
    )
    plot_top_grid_confusion_matrix_rates(
        summary_df,
        output_dir=summary_dir,
    )

    # aggregate summary by model, feature_type, scale and config, and calculate mean score across subjects
    summary_df_aggregated_config = create_aggregated_feature_config_summary(summary_df)
    summary_df_aggregated_config.head(10)
    # save the summary df as csv
    summary_csv_path = summary_dir / "classification_feature-config_summary_aggregated.csv"
    summary_df_aggregated_config.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated feature-config summary CSV to {summary_csv_path.resolve()}")
    # aggregate summary by model, feature_type and scale, and calculate mean score across subjects
    summary_df_aggregated = create_aggregated_feature_summary(summary_df)
    summary_df_aggregated.head(10)
    # save the summary df as csv
    summary_csv_path = summary_dir / "classification_feature_summary_aggregated.csv"
    summary_df_aggregated.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated summary CSV to {summary_csv_path.resolve()}")
    plot_summary_table(
        summary_df_aggregated,
        output_dir=summary_dir,
        plotname="table_classification_feature_summary_aggregated_table",
        title="Grid-Search Model-Feature Summary",
        sort_by="score",
        ascending=False,
        height=800
    )

    # summary plots
    # filter summary_df to only have models: random_forest, svc, logistic_regression
    summary_df_aggregated = summary_df_aggregated[summary_df_aggregated["model_name"].isin(["random_forest", "svc", "logistic_regression"])]
    summary_df_aggregated_config = summary_df_aggregated_config[summary_df_aggregated_config["model_name"].isin(["random_forest", "svc", "logistic_regression"])]

    plot_all_models_average_score(
        summary_df_aggregated,
        output_dir=summary_dir
    )
    plot_mean_sd_plot(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_top_fit_per_feature(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_mean_accuracy_per_feature_all_models(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_violin_per_feature(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_violin_per_model(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_violin_per_model_and_feature(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_violin_per_model_with_feature_scatter(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )
    plot_violin_per_model_with_feature_error_bars(
        summary_df_aggregated_config,
        output_dir=summary_dir
    )

    for model_name in summary_df_aggregated_config["model_name"].unique():
        plot_mean_accuracy_per_feature(
            summary_df_aggregated_config,
            output_dir=summary_dir,
            model_name=model_name
        )

def summarise_kfold_results():

    model_name = "random_forest"
    kfold_model_dir = kfold_dir / model_name

    kfold_summary_path = summary_dir / "classification_kfold_summary.csv"

    if not kfold_summary_path.exists():
        load_all_kfold_results_into_dataframe(kfold_model_dir)
        kfold_df = pd.read_csv(kfold_summary_path)
    else:
        print(f"Kfold summary CSV already exists at {kfold_summary_path.resolve()}, loading it...")
        kfold_df = pd.read_csv(kfold_summary_path)

    kfold_summary_df = create_aggregated_kfold_summary(kfold_df)

    # save the summary df as csv
    summary_csv_path = summary_dir / "classification_kfold_summary_aggregated.csv"
    kfold_summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated kfold summary CSV to {summary_csv_path.resolve()}")
    plot_summary_table(
        kfold_summary_df,
        output_dir=summary_dir,
        plotname="table_classification_kfold_summary_aggregated_table",
        title="Top Random Forest Model K-Fold Summary",
        sort_by="accuracy",
        ascending=False,
        percent_columns=["proportion"],
        height=700
    )

    plot_mean_kfold_accuracy_per_proportion(
        kfold_summary_df,
        output_dir=summary_dir,
        model_name=model_name,
    )
    plot_mean_train_pred_time_per_proportion(
        kfold_summary_df,
        output_dir=summary_dir,
        model_name=model_name,
        time_type="train"
    )
    plot_mean_train_pred_time_per_proportion(
        kfold_summary_df,
        output_dir=summary_dir,
        model_name=model_name,
        time_type="pred"
    )
    plot_kfold_summary_subplots(
        kfold_summary_df,
        output_dir=summary_dir,
        model_name=model_name,
    )

if __name__ == "__main__":
    summarise_results()
    summarise_kfold_results()
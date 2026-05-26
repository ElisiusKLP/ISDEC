import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd


    return (pd,)


@app.cell
def _(pd):
    path = "/Users/peli/Projects/Repositories/ISDEC/results/summary/classification_summary.csv"

    df_subject = pd.read_csv(path)

    df_subject.columns
    return (df_subject,)


@app.cell
def _(df_subject):
    # length of rows
    print(f"Total number of rows: {len(df_subject)}")

    # length of rows for each model
    print(f"Number of rows for each model:\n{df_subject['model_name'].value_counts()}")

    # length of rows for each feature_type
    print(f"Number of rows for each feature type:\n{df_subject['feature_type'].value_counts()}")
    return


@app.cell
def _(pd):
    df_grid_path = "/Users/peli/Projects/Repositories/ISDEC/results/summary/classification_summary_grid.csv"
    df_grid = pd.read_csv(df_grid_path)
    print(f"Total number of rows in grid search summary: {len(df_grid)}")
    print(f"Number of rows for each model in grid search summary:\n{df_grid['model_name'].value_counts()}")
    print(f"Number of rows for each feature type in grid search summary:\n{df_grid['feature_type'].value_counts()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

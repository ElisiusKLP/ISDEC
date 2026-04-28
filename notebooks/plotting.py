import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating some nice plots
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path
    import os

    return Path, mpimg, os, plt


@app.cell
def _(Path, os):
    # set workdir
    print("Current working directory:", Path.cwd())
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
        print("New working directory:", Path.cwd())
    rootdir = Path.cwd()
    return


@app.cell
def _(Path, mpimg, plt):

    img_dir = Path("results/plots/accuracy")
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        language = "en"
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
    out_pdf = img_dir / "all_confusion_matrices_grid.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf.resolve(), bbox_inches="tight")
    print(f"Saved grid PDF to {out_pdf.resolve()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

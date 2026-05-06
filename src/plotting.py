from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
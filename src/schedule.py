import click
from itertools import product

import typer
from rich import print

from classification import FEATURE_CHOICES, MODEL_CHOICES, fit_model, get_model_strategy, train_dir, val_dir

RUN_MODEL_CHOICES = ["svm"]
RUN_FEATURE_CHOICES = ["bandpower_mean", "tfr_morlet", "tfr_dwt_cmor", "tfr_pca"]
RUN_SCALE_CHOICES = ["scale", "no_scale"]

def main(
	model: list[str] = typer.Option(
		list(MODEL_CHOICES),
		"--model",
		"-m",
		help="Models to run.",
		click_type=click.Choice(MODEL_CHOICES),
	),
	feature: list[str] = typer.Option(
		list(FEATURE_CHOICES),
		"--feature",
		"-f",
		help="Feature types to run.",
		click_type=click.Choice(FEATURE_CHOICES),
	),
	scale_state: list[str] = typer.Option(
		["scale", "no_scale"],
		"--scale-state",
		help="Scale states to run.",
		click_type=click.Choice(("scale", "no_scale")),
	),
):
    
    # Helper to override choices if user provides specific options, otherwise use all.
    def override_choices(new_choices: list[str], valid_choices: list[str]):
        if new_choices:
            invalid = [c for c in new_choices if c not in valid_choices]
            if invalid:
                raise ValueError(f"Invalid choices: {invalid}. Valid options: {valid_choices}")
            return new_choices
        return valid_choices

    model = override_choices(RUN_MODEL_CHOICES, MODEL_CHOICES)
    feature = override_choices(RUN_FEATURE_CHOICES, FEATURE_CHOICES)
    scale_state = override_choices(RUN_SCALE_CHOICES, ["scale", "no_scale"])
    
    runs = list(product(model, feature, scale_state))
    for index, (model_name, feature_name, state) in enumerate(runs, start=1):
        scale = state == "scale"
        print(
			f"[{index}/{len(runs)}] model={model_name} feature={feature_name} scale={state}"
        )
        strategy = get_model_strategy(model_name, scale=scale, feature_type=feature_name)
        fit_model(strategy, train_dir, val_dir)

if __name__ == "__main__":
	typer.run(main)

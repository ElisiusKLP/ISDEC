import click
from itertools import product
from pathlib import Path
import json
import typer
from rich import print

from classification import FEATURE_CHOICES, MODEL_CHOICES, fit_model, get_model_strategy, train_dir, val_dir
from summarise_results import summarise_results

# Default schedules file (relative to this module)
SCHEDULES_PATH = Path(__file__).resolve().parent / "schedules.json"

# fallback hardcoded schedules for backwards compatibility
def _load_schedules() -> dict:
	if SCHEDULES_PATH.exists():
		try:
			return json.loads(SCHEDULES_PATH.read_text())
		except Exception:
			raise ValueError(f"Warning: failed to parse {SCHEDULES_PATH}. Please ensure it is valid JSON.")
	else:
		raise ValueError(f"Schedules file not found at {SCHEDULES_PATH}. Please ensure it exists and is valid JSON.")

SCHEDULES = _load_schedules()

def expand_config_grid(model_config: dict[str, object] | None) -> list[dict[str, object]]:
	"""Expand a model config into one config per combination of list-valued parameters."""
	if not model_config:
		return [{}]

	config_keys = list(model_config.keys())
	value_options: list[list[object]] = []
	for key in config_keys:
		value = model_config[key]
		if isinstance(value, list):
			if not value:
				raise ValueError(f"Configuration parameter '{key}' cannot be an empty list")
			value_options.append(list(value))
		else:
			value_options.append([value])

	grid_configs: list[dict[str, object]] = []
	for combo in product(*value_options):
		grid_configs.append(dict(zip(config_keys, combo)))

	return grid_configs

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
	use_config_grid: bool = typer.Option(
		False,
		"--use-config",
		help="Whether to use configuration grid.",
	),
	schedule: int | None = typer.Option(
		None,
		"--schedule",
		"-s",
		help="Schedule number to run. Overrides other options.",
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

	# resolve schedule from schedules.json
	if schedule is None:
		raise ValueError("Please specify a schedule number using --schedule. Available schedules: " + ", ".join(SCHEDULES.keys()))
	schedule_key = str(schedule)
	if schedule_key not in SCHEDULES:
		raise ValueError(f"Schedule '{schedule}' not found in schedules.json. Available: {list(SCHEDULES.keys())}")
	schedule_entry = SCHEDULES[schedule_key]

	RUN_MODEL_CHOICES = schedule_entry.get("models", [])
	RUN_FEATURE_CHOICES = schedule_entry.get("features", [])
	RUN_SCALE_CHOICES = schedule_entry.get("scales", ["scale"]) if schedule_entry else ["scale"]

	model = override_choices(RUN_MODEL_CHOICES, list(MODEL_CHOICES))
	feature = override_choices(RUN_FEATURE_CHOICES, list(FEATURE_CHOICES))
	scale_state = override_choices(RUN_SCALE_CHOICES, ["scale", "no_scale"])

	config_path = Path("src/schedule_config.json").resolve()
	with open(config_path, "r") as f:
		config = json.load(f)

	runs = list(product(model, feature, scale_state))
	for index, (model_name, feature_name, state) in enumerate(runs, start=1):
		scale = state == "scale"
		print(f"[{index}/{len(runs)}] model={model_name} feature={feature_name} scale={state}")

		model_config = config.get(model_name)
		config_grid = expand_config_grid(model_config)
		if model_config:
			print(f"Using configuration grid for {model_name}: {model_config}")
		else:
			print(
				f"No specific configuration found for {model_name} in schedule_config.json. Using defaults"
			)

		for config_index, model_config_variant in enumerate(config_grid, start=1):
			if len(config_grid) > 1:
				print(f"  config [{config_index}/{len(config_grid)}]: {model_config_variant}")

			strategy = get_model_strategy(
				model_name,
				scale=scale,
				feature_type=feature_name,
				config=model_config_variant,
			)
			fit_model(strategy, train_dir, val_dir)


if __name__ == "__main__":
	typer.run(main)
	
	summarise_results()

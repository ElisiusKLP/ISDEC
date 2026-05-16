import click
from itertools import product
from pathlib import Path
import json
import typer
from rich import print
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

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


def _run_task(
	model_name: str, 
	feature_name: str, scale: bool, 
	model_config_variant: dict[str, object], 
	run_idx: int, total_runs: int, 
	config_idx: int, total_configs: int,
	n_jobs_subject: int = 1
	):
	"""Helper to run a single scheduled job. Separated so it can be parallelized with joblib."""
	print(f"[{run_idx}/{total_runs}] model={model_name} feature={feature_name} scale={'scale' if scale else 'no_scale'} config_index={config_idx}/{total_configs}")
	strategy = get_model_strategy(
		model_name,
		scale=scale,
		feature_type=feature_name,
		config=model_config_variant,
	)
	fit_model(strategy, train_dir, val_dir, n_jobs=n_jobs_subject)
	return True

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
	n_jobs: int = typer.Option(
		1,
		"--n-jobs",
		help="Number of parallel jobs to run using joblib.Parallel (default=1).",
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
	# Build task list (model, feature, scale, variant, run_idx, total_runs, config_idx, total_configs)
	tasks: list[tuple] = []
	for run_idx, (model_name, feature_name, state) in enumerate(runs, start=1):
		scale = state == "scale"
		model_config = config.get(model_name)
		if use_config_grid and model_config:
			grid = list(ParameterGrid(model_config))
			print(f"Using configuration grid for {model_name}: {model_config}")
		elif use_config_grid and not model_config:
			grid = [{}]
			print(f"No specific configuration found for {model_name} in schedule_config.json. Using defaults")
		else:
			grid = [{}]
			if model_config:
				print(f"Config exists for {model_name} but --use-config is disabled. Running defaults only")
			else:
				print(f"Running defaults for {model_name} (no config grid)")

		total_configs = len(grid)
		for config_idx, variant in enumerate(grid, start=1):
			tasks.append((model_name, feature_name, scale, variant, run_idx, len(runs), config_idx, total_configs, n_jobs))

	total_tasks = len(tasks)
	print(f"Prepared {total_tasks} tasks (models x features x scales x configs). Running with n_jobs={n_jobs}")

	# run tasks sequentially, parrallel over subjects within each task
	for args in tasks:
		_run_task(*args)

	# TODO: if we wanted to parallelize across tasks instead of subjects, we could do:
	#Parallel(n_jobs=n_jobs)(delayed(_run_task)(*args) for args in tasks)


if __name__ == "__main__":
	typer.run(main)
	
	summarise_results()

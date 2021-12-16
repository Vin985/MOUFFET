import ast
import shutil
from pathlib import Path

import pandas as pd
from mouffet import common_utils, file_utils

DEFAULTS = {
    "run_dir": "config/runs",
    "dest_dir": "results/runs",
    "log_dir": "logs/runs",
    "training_config": "training_config.yaml",
    "evaluation_config": "evaluation_config.yaml",
    "data_config": "data_config.yaml",
    "models_dir": "models",
    "predictions_dir": "predictions",
    "evaluations_dir": "evaluations",
}


def launch_runs(args, handler_classes):
    for run in args.runs:
        launch_run(run, args, handler_classes)


def launch_run(run, args, handler_classes):
    opts_path = Path(args.run_dir) / run
    dest_dir = Path(args.dest_dir) / run
    log_dir = Path(args.log_dir) / run
    model_dir = dest_dir / DEFAULTS["models_dir"]
    evaluation_dir = dest_dir / DEFAULTS["evaluations_dir"]
    predictions_dir = dest_dir / DEFAULTS["predictions_dir"]

    # TODO: Implement clean argument and check if other arguments work
    if args.clean:
        if dest_dir.exists():
            common_utils.print_warning("Removing {}".format(dest_dir))
            shutil.rmtree(dest_dir)
        if log_dir.exists():
            common_utils.print_warning("Removing {}".format(log_dir))
            shutil.rmtree(log_dir)

    # * Perform training
    trainer = handler_classes["training"](
        opts_path=opts_path / args.training_config,
        dh_class=handler_classes["data"],
    )
    for training_scenario in trainer.scenarios:
        # * Make sure all models and logs are saved at the same place
        training_scenario["model_dir"] = str(model_dir)
        training_scenario["logs"]["log_dir"] = str(log_dir)

        # * Data config could be overloaded by model so do not force it
        if not "data_config" in training_scenario:
            training_scenario["data_config"] = str(opts_path / args.data_config)

        trainer.train_scenario(training_scenario)

    # *#####################
    # * Perform evaluation
    # *#####################

    evaluation_config = file_utils.load_config(opts_path / args.evaluation_config)
    # * Make sure predictions and evaluations are saved in the results directory
    evaluation_config["predictions_dir"] = str(predictions_dir)
    evaluation_config["evaluation_dir"] = str(evaluation_dir)

    # * Data config could be overloaded by model so do not force it
    if not "data_config" in evaluation_config:
        evaluation_config["data_config"] = str(opts_path / args.data_config)

    models_stats_path = Path(
        model_dir / handler_classes["training"].MODELS_STATS_FILE_NAME
    )
    models_stats = None
    if models_stats_path.exists():
        models_stats = pd.read_csv(models_stats_path).drop_duplicates(
            "opts", keep="last"
        )
    if models_stats is not None:
        models = [ast.literal_eval(row.opts) for row in models_stats.itertuples()]
        evaluation_config["models"] = models
        evaluator = handler_classes["evaluation"](
            opts=evaluation_config, dh_class=handler_classes["data"]
        )
        evaluator.evaluate()
    else:
        common_utils.print_error(
            "No trained models found for this run. Please train models before evaluating them!"
        )

import ast
import shutil
from pathlib import Path

import pandas as pd

from mouffet import common_utils, config_utils, file_utils

from .parser import RunArgumentParser


class RunHandler:

    DEFAULT_RUN_ARGS = {
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

    def __init__(self, handler_classes, *args, default_args=None, **kwargs) -> None:
        self.handler_classes = handler_classes
        self.defaults = self.DEFAULT_RUN_ARGS
        if default_args is not None:
            self.defaults.update(default_args)
        self.parser = RunArgumentParser(*args, default_args=self.defaults, **kwargs)
        self.args = self.parser.parse_args()

    def launch_runs(self):
        for run in self.args.runs:
            self.launch_run(run)

    def clean_model_options(self, models):
        for model in models:
            if "weights_opts" in model:
                # * Remove information about weight loading used in training for the evaluation as
                # * it prevents the new weights from being loaded
                model["weights_opts"] = {}
        return models

    def launch_run(self, run):
        opts_path = Path(self.args.run_dir) / run
        dest_dir = Path(self.args.dest_dir) / run
        log_dir = Path(self.args.log_dir) / run
        model_dir = dest_dir / self.defaults["models_dir"]
        evaluation_dir = dest_dir / self.defaults["evaluations_dir"]
        predictions_dir = dest_dir / self.defaults["predictions_dir"]

        # TODO: Implement clean argument and check if other arguments work
        if self.args.clean:
            if dest_dir.exists():
                common_utils.print_warning("Removing {}".format(dest_dir))
                shutil.rmtree(dest_dir)
            if log_dir.exists():
                common_utils.print_warning("Removing {}".format(log_dir))
                shutil.rmtree(log_dir)

        # * Perform training
        trainer = self.handler_classes["training"](
            opts_path=opts_path / self.args.training_config,
            dh_class=self.handler_classes["data"],
        )
        for training_scenario in trainer.scenarios:
            # * Make sure all models and logs are saved at the same place
            training_scenario["model_dir"] = str(model_dir)
            if not "logs" in training_scenario:
                training_scenario["logs"] = {}
            training_scenario["logs"]["log_dir"] = str(log_dir)

            # * Data config could be overloaded by model so do not force it
            if not "data_config" in training_scenario:
                training_scenario["data_config"] = str(
                    opts_path / self.args.data_config
                )

            trainer.train_scenario(training_scenario)

        # *#####################
        # * Perform evaluation
        # *#####################

        evaluation_config = file_utils.load_config(
            opts_path / self.args.evaluation_config
        )
        # * Make sure predictions and evaluations are saved in the results directory
        evaluation_config["predictions_dir"] = str(predictions_dir)
        evaluation_config["evaluation_dir"] = str(evaluation_dir)

        # * Data config could be overloaded by model so do not force it
        if "data_config" not in evaluation_config:
            evaluation_config["data_config"] = str(opts_path / self.args.data_config)

        models_stats_path = Path(model_dir) / config_utils.MODELS_STATS_FILE_NAME
        models_stats = None
        if models_stats_path.exists():
            models_stats = pd.read_csv(models_stats_path).drop_duplicates(
                "opts", keep="last"
            )
        if models_stats is not None:
            models = [ast.literal_eval(row.opts) for row in models_stats.itertuples()]
            models = self.clean_model_options(models)
            evaluation_config["models"] = models
            evaluator = self.handler_classes["evaluation"](
                opts=evaluation_config, dh_class=self.handler_classes["data"]
            )
            evaluator.evaluate()
        else:
            common_utils.print_error(
                "No trained models found for this run. Please train models before evaluating them!"
            )

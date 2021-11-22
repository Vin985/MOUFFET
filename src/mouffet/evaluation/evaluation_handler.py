import time
import traceback
from abc import abstractmethod
from datetime import datetime
from itertools import product
from pathlib import Path

import feather
import pandas as pd
from mouffet.options.model_options import ModelOptions

from ..plotting import plot
from ..utils import common as common_utils
from ..utils import file as file_utils
from ..utils.model_handler import ModelHandler


class EvaluationHandler(ModelHandler):
    """Base class for evaluating models. Inherits ModelHandler

    Relevant options:

    <Global>
    data_config: Path to the data configuration file used to initialize the data handler.
    evaluation_dir: Directory where to save results

    <Models>
    model_dir: Directory where to load models.
    predictions_dir: Directory where to load/save predictions.
    reclassify: Run the model again even if a prediction file is found

    <Evaluators>
    type: The type of evaluator corresponding to one of the keys of EVALUATORS attribute of the subclass


    Args:
        ModelHandler ([type]): [description]

    Raises:
        AttributeError: [description]

    Returns:
        [type]: [description]
    """

    PREDICTIONS_STATS_FILE_NAME = "predictions_stats.csv"

    EVALUATORS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        plot.set_plotting_method(self.opts)  # pylint: disable=no-member

    @staticmethod
    @abstractmethod
    def classify_element(model, element, *args, **kwargs):
        pass

    @abstractmethod
    def classify_test_data(self, model, database):
        pass

    def get_predictions_dir(self, model_opts, database):
        preds_dir = self.get_option("predictions_dir", model_opts)
        if not preds_dir:
            raise AttributeError(
                "Please provide a directory where to save the predictions using"
                + " the predictions_dir option in the config file"
            )
        return Path(preds_dir)

    def get_predictions_file_name(self, model_opts, database):
        return (
            database.name
            + "_"
            + model_opts.model_id
            + "_v"
            + str(model_opts.load_version)
            + ".feather"
        )

    def get_predictions(self, model_opts, database):
        preds_dir = self.get_predictions_dir(model_opts, database)
        file_name = self.get_predictions_file_name(model_opts, database)
        pred_file = preds_dir / file_name
        if not model_opts.get("reclassify", False) and pred_file.exists():
            predictions = feather.read_dataframe(pred_file)
        else:
            # * Load predictions stats database
            scenario_info = {}
            preds_stats = None
            preds_stats_dir = Path(self.get_option("predictions_dir", model_opts))
            preds_stats_path = preds_stats_dir / self.PREDICTIONS_STATS_FILE_NAME
            if preds_stats_path.exists():
                preds_stats = pd.read_csv(preds_stats_path)
            model_opts.opts["data_config"] = self.opts["data_config"]
            model_opts.opts["model_dir"] = self.get_option("model_dir", model_opts)
            model_opts.opts["inference"] = True
            common_utils.print_info("Loading model with options: " + str(model_opts))
            model = self.load_model(model_opts)
            predictions, infos = self.classify_test_data(model, database)

            # * save classification stats
            scenario_info["date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            scenario_info["model_id"] = model_opts.model_id
            # scenario_info["database"] =
            # scenario_info["classification_duration"] = round(end - start, 2)
            # scenario_info["n_files"] =
            # scenario_info["opts"] = str(scenario)
            scenario_info.update(infos)

            df = pd.DataFrame([scenario_info])
            if preds_stats is not None:
                preds_stats = pd.concat([preds_stats, df])
                preds_stats = preds_stats.drop_duplicates(
                    subset=["database", "model_id"], keep="last"
                )
            else:
                preds_stats = df
            file_utils.ensure_path_exists(preds_stats_path, is_file=True)
            preds_stats.to_csv(preds_stats_path, index=False)

            pred_file.parent.mkdir(parents=True, exist_ok=True)
            feather.write_dataframe(predictions, pred_file)
        return predictions

    def get_evaluator(self, evaluator_opts):
        evaluator = self.EVALUATORS.get(evaluator_opts["type"], None)
        if not evaluator:
            print(
                "Evaluator {} not found. Please make sure this evaluator exists."
                + "Skipping."
            )
            return None
        return evaluator

    def consolidate_results(self, results):
        res = common_utils.listdict2dictlist(results)
        res["matches"] = pd.concat(res["matches"])
        res["stats"] = pd.concat(res["stats"])
        res["plots"] = common_utils.listdict2dictlist(
            res.get("plots", []), flatten=True
        )
        return res

    def save_pr_curve_data(self, pr_df):
        print("saving_pr_curve data")
        res_dir = Path(self.opts.get("evaluation_dir", "."))
        pr_file = res_dir / self.opts.get("PR_curve_save_file", "PR_curves.feather")
        if pr_file.exists():
            pr_curves = pd.read_feather(pr_file)
            res = pd.concat([pr_df, pr_curves]).drop_duplicates()
        else:
            res = pr_df

        res.reset_index(inplace=True, drop=True)
        res.to_feather(pr_file)

    def save_results(self, results):
        res = self.consolidate_results(results)
        time = datetime.now()
        res_dir = Path(self.opts.get("evaluation_dir", ".")) / time.strftime("%Y%m%d")
        prefix = time.strftime("%H%M%S")
        eval_id = self.opts.get("id", "")
        res["stats"].to_csv(
            str(
                file_utils.ensure_path_exists(
                    res_dir / ("_".join(filter(None, [prefix, eval_id, "stats.csv"]))),
                    is_file=True,
                )
            ),
            index=False,
        )

        pr_df = res["stats"].loc[res["stats"]["PR_curve"] == True]

        if not pr_df.empty:
            self.save_pr_curve_data(pr_df)

        plots = res.get("plots", {})
        if plots:
            for key, values in plots.items():
                # pylint: disable=no-member
                plot.save_as_pdf(
                    values,
                    res_dir
                    / (
                        "_".join(
                            filter(
                                None,
                                [prefix, eval_id, "{}.pdf".format(key)],
                            )
                        )
                    ),
                )

    def expand_scenarios(self, element_type):
        elements = self.opts[element_type]
        default = self.opts.get(element_type + "_options", {})
        scenarios = []
        for element in elements:
            # * Add default options to scenario
            tmp = common_utils.deep_dict_update(default, element, copy=True)
            if "scenarios" in tmp:
                scenario = tmp.pop("scenarios")
                for opts in common_utils.expand_options_dict(scenario):
                    # * Add expanded options
                    res = common_utils.deep_dict_update(tmp, opts, copy=True)
                    scenarios.append(res)
            else:
                scenarios.append(tmp)
        return scenarios

    def load_scenarios(self):
        db_scenarios = self.expand_scenarios("databases")
        model_scenarios = self.expand_scenarios("models")
        evaluator_scenarios = self.expand_scenarios("evaluators")
        res = product(db_scenarios, model_scenarios, evaluator_scenarios)
        return list(res)

    def load_tags(self, database, types):
        return self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": types}
        )

    def add_global_options(self, opts):
        if "models_options" in self.opts:
            opts.add_option(
                "models_options", self.opts["models_options"], overwrite=False
            )
        if "databases_options" in self.opts:
            opts.add_option(
                "databases_options", self.opts["databases_options"], overwrite=False
            )
        return opts

    def skip_database(self, db, evaluator_opts):
        include = evaluator_opts.get("databases", [])
        if include and not db in include:
            common_utils.print_info(
                "Database {} is not in the accepted databases list of evaluator {}. Skipping.".format(
                    db, evaluator_opts["type"]
                )
            )
            return True
        exclude = evaluator_opts.get("exclude_databases", [])
        if exclude and db in exclude:
            common_utils.print_info(
                "Database {} is in the excluded databases of evaluator {}. Skipping.".format(
                    db, evaluator_opts["type"]
                )
            )
            return True
        return False

    def evaluate_scenario(self, opts):
        try:
            db_opts, model_opts, evaluator_opts = opts
            if self.skip_database(db_opts["name"], evaluator_opts):
                return {}

            model_opts = ModelOptions(model_opts)
            # * Add global option to model options for id resolution
            model_opts = self.add_global_options(model_opts)
            if "databases_options" in model_opts:
                common_utils.deep_dict_update(db_opts, model_opts.databases_options)

            # * Duplicate database options
            database = self.data_handler.duplicate_database(db_opts)
            model_stats = {}
            if database and database.has_type("test"):
                self.data_handler.check_dataset(database, ["test"])
                preds = self.get_predictions(model_opts, database)
                preds = preds.rename(columns={"recording_path": "recording_id"})
                stats_infos = {
                    "database": database.name,
                    "model": model_opts.model_id,
                    "class": database.class_type,
                }
                stats_opts = {
                    "database_opts": str(database.updated_opts),
                    "model_opts": str(model_opts),
                }
                print(
                    "\033[92m"
                    + "Evaluating model {0} on test dataset {1} with evaluator {2}".format(
                        model_opts.model_id, database.name, evaluator_opts["type"]
                    )
                    + "\033[0m"
                )
                evaluator_opts["scenario_info"] = stats_infos
                evaluator = self.get_evaluator(evaluator_opts)
                if evaluator:
                    tags = self.load_tags(database, evaluator.REQUIRES)
                    start = time.time()
                    model_stats = evaluator.run_evaluation(preds, tags, evaluator_opts)
                    end = time.time()
                    model_stats["stats"] = pd.concat(
                        [
                            pd.DataFrame([stats_infos]),
                            model_stats["stats"].assign(**stats_opts),
                        ],
                        axis=1,
                    )
                    model_stats["stats"]["PR_curve"] = evaluator_opts.get(
                        "do_PR_curve", False
                    )
                    model_stats["stats"]["duration"] = round(end - start, 2)
                    model_stats["stats"]["evaluator"] = evaluator_opts["type"]

                    return model_stats
        except Exception:
            print(traceback.format_exc())
            common_utils.print_error(
                "Error evaluating the model for scenario {}".format(opts)
            )
            return {}

    def evaluate(self):
        stats = [self.evaluate_scenario(scenario) for scenario in self.scenarios]
        if self.opts.get("save_results", True):
            self.save_results(stats)
        return stats

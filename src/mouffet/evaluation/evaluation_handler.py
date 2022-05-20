import time
import traceback
from abc import abstractmethod
from datetime import datetime
from itertools import product
from pathlib import Path

import feather
import pandas as pd

from ..options import ModelOptions
from ..plotting import plot
from ..utils import ModelHandler, common_utils, file_utils
from . import EVALUATORS


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
    type: The type of evaluator corresponding to one of the keys of EVALUATORS attribute
    of the subclass


    Args:
        ModelHandler ([type]): [description]

    Raises:
        AttributeError: [description]

    Returns:
        [type]: [description]
    """

    PREDICTIONS_STATS_FILE_NAME = "predictions_stats.csv"

    # EVALUATORS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        plot.set_plotting_method(self.opts)  # pylint: disable=no-member

    @abstractmethod
    def classify_database(self, model, database, db_type="test"):
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

    def on_get_predictions_end(self, preds):
        return preds

    def get_predictions(self, model_opts, database):
        preds_dir = self.get_predictions_dir(model_opts, database)
        file_name = self.get_predictions_file_name(model_opts, database)
        pred_file = preds_dir / file_name
        if not model_opts.get("reclassify", False) and pred_file.exists():
            preds = feather.read_dataframe(pred_file)
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
            preds, infos = self.classify_database(model, database, db_type="test")

            # * save classification stats
            scenario_info["date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            scenario_info["model_id"] = model_opts.model_id
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
            feather.write_dataframe(preds, pred_file)
        preds = self.on_get_predictions_end(preds)
        return preds

    # def get_evaluator(self, evaluator_opts):
    #     evaluator = self.EVALUATORS.get(evaluator_opts["type"], None)
    #     if not evaluator:
    #         print(
    #             "Evaluator {} not found. Please make sure this evaluator exists."
    #             + "Skipping."
    #         )
    #         return None
    #     return evaluator

    def consolidate_results(self, results):
        res = common_utils.listdict2dictlist(results)
        if res:
            if "matches" in res:
                res["matches"] = pd.concat(res["matches"])
            if "stats" in res:
                res["stats"] = pd.concat(res["stats"])
            if "plots" in res:
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
        file_names = {}
        if res:
            prefix = ""
            cur_time = datetime.now()
            res_dir = Path(self.opts.get("evaluation_dir", "."))

            if self.opts.get("save_use_date_subfolder", True):
                res_dir /= cur_time.strftime("%Y%m%d")
            if self.opts.get("save_use_time_prefix", True):
                prefix = cur_time.strftime("%H%M%S")
            eval_id = self.opts.get("id", "")
            stats_file_path = str(
                file_utils.ensure_path_exists(
                    res_dir / ("_".join(filter(None, [prefix, eval_id, "stats.csv"]))),
                    is_file=True,
                )
            )
            res["stats"].to_csv(stats_file_path, index=False)
            file_names["stats"] = stats_file_path
            pr_df = res["stats"].loc[
                res["stats"]["PR_curve"] == True  # pylint: disable=singleton-comparison
            ]

            if not pr_df.empty:
                self.save_pr_curve_data(pr_df)

            plots = res.get("plots", {})
            if plots:
                for key, values in plots.items():
                    # pylint: disable=no-member
                    plot_file_path = res_dir / (
                        "_".join(
                            filter(
                                None,
                                [prefix, eval_id, "{}.pdf".format(key)],
                            )
                        )
                    )
                    plot.save_as_pdf(
                        values,
                        plot_file_path,
                    )
                    file_names["plot_" + key] = plot_file_path
        return file_names

    def expand_scenarios(self, element_type):
        if not element_type in self.opts:
            elements = []
        else:
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

    def get_models_by_id(self):
        model_ids = self.opts.get("model_ids", [])
        for model_id in model_ids:
            if isinstance(model_id, dict):
                pass
            if isinstance(model_id, str):
                pass

        return []

    def get_model_scenarios(self):
        model_scenarios = self.expand_scenarios("models")
        # model_scenarios += self.get_models_by_id()
        return model_scenarios

    def load_scenarios(self):
        db_scenarios = self.expand_scenarios("databases")
        model_scenarios = self.get_model_scenarios()
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

    def perform_evaluation(
        self, evaluator, evaluation_data, scenario_infos, scenario_opts
    ):
        eval_result = {}
        # if self.opts.get("events_only", False):
        #     print(
        #         "\033[92m"
        #         + "Getting events for model {0} on dataset {1} with evaluator {2}".format(
        #             scenario_infos["model"],
        #             scenario_infos["database"],
        #             scenario_infos["evaluator"],
        #         )
        #         + "\033[0m"
        #     )
        #     eval_result["events"] = evaluator.get_events(
        #         evaluation_data, scenario_opts["evaluator_opts"]
        #     )
        #     eval_result["conf"] = dict(scenario_infos, **scenario_opts)
        # else:
        print(
            "\033[92m"
            + "Processing model {0} on dataset {1} with evaluator {2}".format(
                scenario_infos["model"],
                scenario_infos["database"],
                scenario_infos["evaluator"],
            )
            + "\033[0m"
        )

        start = time.time()
        eval_result = evaluator.run_evaluation(
            evaluation_data, scenario_opts["evaluator_opts"], scenario_infos
        )
        end = time.time()
        if eval_result:
            eval_result["stats"] = eval_result.get("stats", {})
            eval_result["stats"]["PR_curve"] = scenario_opts["evaluator_opts"].get(
                "do_PR_curve", False
            )
            eval_result["stats"]["duration"] = round(end - start, 2)

            eval_result["stats"] = pd.concat(
                [
                    pd.DataFrame([scenario_infos]),
                    eval_result["stats"].assign(
                        **{key: str(value) for key, value in scenario_opts.items()}
                    ),
                ],
                axis=1,
            )
        return eval_result

    def get_evaluation_data(self, evaluator, database, model_opts, evaluator_opts):
        self.data_handler.check_dataset(database, ["test"])
        preds = self.get_predictions(model_opts, database)
        if evaluator_opts.get("filter_only", False):
            tags = None
        else:
            tags = self.data_handler.load_dataset(
                database,
                "test",
                load_opts={"file_types": evaluator.REQUIRES},
            )
        return preds, tags

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
            try:
                database = self.data_handler.duplicate_database(db_opts)
            except KeyError:
                common_utils.print_error(
                    (
                        "Database '{}' does not exists. Please check that the "
                        + " database is properly defined in the data configuration file"
                    ).format(db_opts["name"])
                )
                return {}
            eval_result = {}
            if database and database.has_type("test"):
                scenario_infos = {
                    "database": database.name,
                    "model": model_opts.model_id,
                    "class": database.class_type,
                    "evaluator": evaluator_opts.get("type", None),
                    "evaluation_id": self.opts.get("id", ""),
                }

                scenario_opts = {
                    "evaluator_opts": evaluator_opts,
                    "database_opts": database.updated_opts,
                    "model_opts": model_opts,
                }
                evaluator_opts["scenario_info"] = scenario_infos
                evaluator = EVALUATORS[evaluator_opts.get("type", None)]

                if evaluator:
                    evaluation_data = self.get_evaluation_data(
                        evaluator, database, model_opts, evaluator_opts
                    )

                    eval_result = self.perform_evaluation(
                        evaluator, evaluation_data, scenario_infos, scenario_opts
                    )

                    return eval_result
        except Exception:
            print(traceback.format_exc())
            common_utils.print_error(
                "Error evaluating the model for scenario {}".format(opts)
            )
            return {}

    def evaluate(self):
        res = [self.evaluate_scenario(scenario) for scenario in self.scenarios]
        if self.opts.get("save_results", True):
            self.save_results(res)
        return res

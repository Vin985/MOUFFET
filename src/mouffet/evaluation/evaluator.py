from abc import abstractmethod
from datetime import datetime
from itertools import product
from pathlib import Path

import feather
import pandas as pd
import plotnine
from mouffet.options.model_options import ModelOptions
from plotnine.labels import ggtitle

from ..utils import common as common_utils
from ..utils import file as file_utils
from ..utils.model_handler import ModelHandler


class Evaluator(ModelHandler):
    """ Base class for evaluating models. Inherits ModelHandler

    Relevant options:

    <Global>
    data_config: Path to the data configuration file used to initialize the data handler.
    evaluation_dir: Directory where to save results

    <Models>
    model_dir: Directory where to load models.
    predictions_dir: Directory where to load/save predictions.
    reclassify: Run the model again even if a prediction file is found

    <Detectors>
    type: The type of detector corresponding to one of the keys of DETECTORS attribute of the subclass


    Args:
        ModelHandler ([type]): [description]

    Raises:
        AttributeError: [description]

    Returns:
        [type]: [description]
    """

    DETECTORS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

            model_opts.opts["data_config"] = self.opts["data_config"]
            model_opts.opts["model_dir"] = self.get_option("model_dir", model_opts)
            model = self.load_model(model_opts)
            predictions = self.classify_test_data(model, database)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            feather.write_dataframe(predictions, pred_file)
        return predictions

    def get_detector(self, detector_opts):
        detector = self.DETECTORS.get(detector_opts["type"], None)
        if not detector:
            print(
                "Detector {} not found. Please make sure this detector exists."
                + "Skipping."
            )
            return None
        return detector

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

        print(res)
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
        )

        pr_df = res["stats"].loc[res["stats"]["PR_curve"] == True]

        if not pr_df.empty:
            self.save_pr_curve_data(pr_df)

        plots = res.get("plots", {})
        if plots:
            for key, values in plots.items():
                plotnine.save_as_pdf_pages(
                    values,
                    res_dir
                    / (
                        "_".join(filter(None, [prefix, eval_id, "{}.pdf".format(key)],))
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
        detector_scenarios = self.expand_scenarios("detectors")
        res = product(db_scenarios, model_scenarios, detector_scenarios)
        return list(res)

    def load_tags(self, database, types):
        return self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": types}
        )

    def add_global_options(self, opts):
        if "models_options" in self.opts:
            opts.add_option("models_options", self.opts["models_options"])
        if "databases_options" in self.opts:
            opts.add_option("databases_options", self.opts["databases_options"])
        return opts

    def evaluate_scenario(self, opts):
        db_opts, model_opts, detector_opts = opts
        model_opts = ModelOptions(model_opts)
        model_opts = self.add_global_options(model_opts)
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
                + "Evaluating model {0} on test dataset {1}".format(
                    model_opts.name, database.name
                )
                + "\033[0m"
            )
            detector_opts["scenario_info"] = stats_infos
            detector = self.get_detector(detector_opts)
            if detector:
                tags = self.load_tags(database, detector.REQUIRES)
                model_stats = detector.run_evaluation(preds, tags, detector_opts)
                model_stats["stats"] = pd.concat(
                    [
                        pd.DataFrame([stats_infos]),
                        model_stats["stats"].assign(**stats_opts),
                    ],
                    axis=1,
                )
                model_stats["stats"]["PR_curve"] = detector_opts.get(
                    "do_PR_curve", False
                )

        return model_stats

    def evaluate(self):
        stats = [self.evaluate_scenario(scenario) for scenario in self.scenarios]
        if self.opts.get("save_results", True):
            self.save_results(stats)
        return stats


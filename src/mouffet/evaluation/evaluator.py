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

    DETECTORS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self, model_opts):
        version = model_opts.load_version
        old_opts = file_utils.load_config(
            Path(self.get_option("model_dir", model_opts))
            / model_opts["name"]
            / str(version)
            / "network_opts.yaml"
        )
        old_opts["data_config"] = self.opts["data_config"]
        opts = ModelOptions(old_opts)

        # * Problem: to load the model, we need to use the old model id
        # * Or we can define a new one in the config option to save the results which leads in
        # * in inconsistencies.
        # * Solution: use scenario id instead and prevent from overriding model id

        # weights_path = opts.get_weights_path(
        #     epoch=model_opts.get("from_epoch", 0),
        #     version=model_opts.get("from_version", -1),
        # )
        common_utils.deep_dict_update(
            opts.opts, model_opts.opts, except_keys=["id", "id_prefixes"]
        )
        model = self.get_model_instance(opts)
        model.load_weights(from_epoch=opts["model"].get("from_epoch", 0))
        return model

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
            model = self.get_model(model_opts)
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

    def consolidate_stats(self, stats):
        tmp_stats, plots = [], []
        for stat in stats:
            tmp_stats.append(pd.Series(stat["stats"]))
            plt = stat.get("tag_repartition", None)
            if plt:
                plots.append(plt)
        stats_df = pd.DataFrame(tmp_stats)
        return stats_df, plots

    def save_results(self, stats):
        stats_df, plots = self.consolidate_stats(stats)
        time = datetime.now()
        res_dir = Path(self.opts.get("evaluation_dir", ".")) / time.strftime("%Y%m%d")
        prefix = time.strftime("%H%M%S")
        eval_id = self.opts.get("id", "")
        stats_df.to_csv(
            str(
                file_utils.ensure_path_exists(
                    res_dir / ("_".join(filter(None, [prefix, eval_id, "stats.csv"]))),
                    is_file=True,
                )
            ),
        )
        if plots:
            plotnine.save_as_pdf_pages(
                plots,
                res_dir
                / ("_".join(filter(None, [prefix, eval_id, "tag_repartition.pdf"],))),
            )

    def expand_scenarios(self, element_type):
        elements = self.opts[element_type]
        default = self.opts.get(element_type + "_options", {})
        scenarios = []
        for element in elements:
            if "scenarios" in element:
                clean = dict(element)
                clean.pop("scenarios")
                for opts in common_utils.expand_options_dict(element["scenarios"]):
                    # * Add default options to scenario
                    res = common_utils.deep_dict_update(clean, default, copy=True)
                    # * Add expanded options
                    res = common_utils.deep_dict_update(res, opts, copy=True)
                    scenarios.append(res)
            else:
                scenarios.append(
                    common_utils.deep_dict_update(element, default, copy=True)
                )
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

    def evaluate_scenario(self, opts):
        db_opts, model_opts, detector_opts = opts
        model_opts = ModelOptions(model_opts)
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
                "detector_opts": str(detector_opts),
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
            detector = self.get_detector(detector_opts)
            if detector:
                tags = self.load_tags(database, detector.REQUIRES)
                model_stats = detector.evaluate(preds, tags, detector_opts)
                stats_infos.update(model_stats["stats"])
                stats_infos.update(stats_opts)
                model_stats["stats"] = stats_infos
                plt = model_stats.get("tag_repartition", None)
                if plt:
                    plt += ggtitle(
                        (
                            "Tag repartition for model {}, database {}, class {}\n"
                            + "with detector options {}"
                        ).format(
                            model_opts.model_id,
                            database.name,
                            database.class_type,
                            detector_opts,
                        )
                    )
        return model_stats

    def evaluate(self):
        stats = [self.evaluate_scenario(scenario) for scenario in self.scenarios]
        if self.opts.get("save_results", True):
            self.save_results(stats)
        return stats

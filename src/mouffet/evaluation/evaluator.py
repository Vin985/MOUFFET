from abc import ABC, abstractmethod

import pandas as pd

from ..utils import common_utils
from ..plotting import plot


class Evaluator(ABC):

    NAME = ""

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"start": 0, "end": 1, "step": 0.05},
    }

    PLOTS = {}

    REQUIRES = []

    def requires(self, options):
        return self.REQUIRES

    def run_evaluation(self, data, options, infos):
        res = {}
        if options.get("filter_only", False):
            predictions, _ = data
            res["events"] = self.filter_predictions(predictions, options)
            res["stats"] = {}
        elif options.get("do_PR_curve", False):
            res = self.get_PR_curve(data, options, infos)
        else:
            res = self.evaluate_scenario(data, options, infos)
        return res

    def evaluate_scenario(self, data, options, infos):
        res = self.evaluate(data, options, infos)
        return res

    @abstractmethod
    def evaluate(self, data, options, infos):
        return {"stats": None, "matches": None}

    def get_PR_scenarios(self, options):
        pr_scenarios = options.get("scenarios_PR_curve", {})
        if not pr_scenarios:
            common_utils.print_warning(
                "do_PR_curve is set to True but no option for scenarios_PR_curve has been found."
            )
        scenarios = []
        for scenario in common_utils.expand_options_dict(pr_scenarios):
            tmp = common_utils.deep_dict_update(options, scenario, copy=True)
            # options[opts["variable"]] = opts["values"]
            scenarios.append(tmp)
        # scenarios = common_utils.expand_options_dict(options)
        return scenarios

    def get_PR_curve(self, data, options, infos):
        scenarios = self.get_PR_scenarios(options)
        tmp = []
        for scenario in scenarios:
            tmp.append(self.evaluate_scenario(data, scenario, infos))

        res = common_utils.listdict2dictlist(tmp)
        res["matches"] = pd.concat(res["matches"])
        res["stats"] = pd.concat(res["stats"])
        res["plots"] = common_utils.listdict2dictlist(res.get("plots", []))
        if options.get("draw_plots", True):
            res = plot.plot_PR_curve(res, options)  # pylint: disable=no-member
        return res

    def draw_plots(self, data, options, infos):
        res = {}
        plots = options.get("plots", [])
        for to_plot in plots:
            func = self.PLOTS.get(to_plot, None)
            if func is not None:
                tmp = func(data, options, infos)
                res[to_plot] = tmp

            # func_name = "plot_" + to_plot.strip()
            # if hasattr(self, func_name) and callable(getattr(self, func_name)):
            #     tmp = getattr(self, func_name)(data, options, infos)
            #     res[to_plot] = tmp
        return res

    def filter_predictions(self, predictions, options, tags=None):
        return []

    def check_database(self, data, options, infos):
        if infos["database"] not in options.get(self.NAME + "_databases", []):
            common_utils.print_info(
                (
                    "Database {0} is not part of the accepted databases for the '{1}' "
                    + "evaluator described in the '{1}_databases' option. Skipping."
                ).format(options["scenario_info"]["database"], self.NAME)
            )
            return False
        return True

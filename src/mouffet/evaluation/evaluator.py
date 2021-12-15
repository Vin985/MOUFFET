from abc import ABC, abstractmethod

import pandas as pd

from ..utils import common_utils
from ..plotting import plot


class Evaluator(ABC):

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"start": 0, "end": 1, "step": 0.05},
    }

    DEFAULT_PLOTS = []

    def run_evaluation(self, predictions, tags, options):
        if options.get("do_PR_curve", False):
            return self.get_PR_curve(predictions, tags, options)
        else:
            return self.evaluate_scenario(predictions, tags, options)

    def evaluate_scenario(self, predictions, tags, options):
        res = self.evaluate(predictions, tags, options)
        res["stats"]["options"] = str(options)
        return res

    @abstractmethod
    def evaluate(self, predictions, tags, options):
        return {"stats": None, "matches": None}

    def get_PR_scenarios(self, options):
        opts = common_utils.deep_dict_update(
            self.DEFAULT_PR_CURVE_OPTIONS, options.pop("PR_curve", {})
        )
        options[opts["variable"]] = opts["values"]
        scenarios = common_utils.expand_options_dict(options)
        return scenarios

    def get_PR_curve(self, predictions, tags, options):
        scenarios = self.get_PR_scenarios(options)
        tmp = []
        for scenario in scenarios:
            tmp.append(self.evaluate_scenario(predictions, tags, scenario))

        res = common_utils.listdict2dictlist(tmp)
        res["matches"] = pd.concat(res["matches"])
        res["stats"] = pd.concat(res["stats"])
        res["plots"] = common_utils.listdict2dictlist(res.get("plots", []))
        if options.get("draw_plots", True):
            res = plot.plot_PR_curve(res, options)  # pylint: disable=no-member
        return res

    def draw_plots(self, data, options):
        res = {}
        plots = options.get("plots", self.DEFAULT_PLOTS)
        for to_plot in plots:
            func_name = "plot_" + to_plot.strip()
            if hasattr(self, func_name) and callable(getattr(self, func_name)):
                tmp = getattr(self, func_name)(data, options)
                res[to_plot] = tmp
        return res

    @abstractmethod
    def get_events(self, predictions, options, *args, **kwargs):
        return []

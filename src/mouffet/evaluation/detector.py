from abc import ABC, abstractmethod

import pandas as pd

from ..utils.common import deep_dict_update, expand_options_dict


class Detector(ABC):

    EVENTS_COLUMNS = {
        "index": "event_id",
        "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    DEFAULT_MIN_ACTIVITY = 0.85
    DEFAULT_MIN_DURATION = 0.1
    DEFAULT_END_THRESHOLD = 0.6

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"end": 1, "start": 0, "step": 0.05},
    }

    def __init__(self):
        pass

    @abstractmethod
    def get_events(self, predictions, options, *args, **kwargs):
        pass

    def evaluate(self, predictions, tags, options):
        if options.get("do_PR_curve", False):
            return self.get_PR_curve(predictions, tags, options)
        else:
            return self.evaluate_scenario(predictions, tags, options)

    @abstractmethod
    def evaluate_scenario(self, predictions, tags, options):
        return {"options": options, "stats": [], "matches": []}

    def get_PR_scenarios(self, options):
        opts = deep_dict_update(
            self.DEFAULT_PR_CURVE_OPTIONS, options.pop("PR_curve", {})
        )
        options[opts["variable"]] = opts["values"]
        scenarios = expand_options_dict(options)
        return scenarios

    def get_PR_curve(self, predictions, tags, options):
        scenarios = self.get_PR_scenarios(options)
        tmp = []
        for scenario in scenarios:
            tmp.append(self.evaluate_scenario(predictions, tags, scenario))

        res = {
            k: [d.get(k) for d in tmp]
            for k in {key for tmp_dict in tmp for key in tmp_dict}
        }
        res["matches"] = pd.concat(res["matches"])
        res["PR_curve"] = pd.DataFrame(res["stats"])
        res["options"] = pd.DataFrame(res["options"])
        if options.get("draw_plots", True):
            res = self.plot_PR_curve(res, options)
        return res

    def draw_plots(self, options, **kwargs):
        return None

    def plot_PR_curve(self, stats, options):
        return {}


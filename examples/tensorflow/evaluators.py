import pandas as pd
from mouffet.evaluation import Evaluator
from mouffet import common_utils
from sklearn.metrics import classification_report
import numpy as np


class TFBasicEvaluator(Evaluator):
    def evaluate(self, data, options, infos):
        # * Just return the data provided by tensorflow
        common_utils.print_warning("Accuracy: {}".format(round(data["accuracy"], 3)))
        return {"stats": pd.DataFrame([data]), "matches": pd.DataFrame()}


class CustomEvaluator(Evaluator):
    def get_label_names(self, columns, metadata):
        res = []
        for x in columns:
            if x == -1:
                res.append("Not sure")
            else:
                res.append(metadata.features["label"].int2str(int(x)))
        return res

    def evaluate(self, data, options, infos):
        preds, labels, meta = data
        # * Get class with the best prediction score
        thresh = options.get("threshold", -1)
        npreds = preds.to_numpy()
        top_class = npreds.argmax(axis=1)
        if thresh != -1:
            unsolved = npreds.max(axis=1) <= thresh
            top_class[unsolved] = -1

        # * Get label names
        label_names = self.get_label_names(np.unique(top_class), meta)
        # * get classification report from sklearn
        cr = classification_report(
            labels,
            top_class,
            output_dict=True,
            target_names=label_names,
        )
        equals = (labels == top_class).astype(int)
        # * Print report
        common_utils.print_warning(
            classification_report(labels, top_class, target_names=label_names)
        )
        return {
            "stats": pd.DataFrame([cr]),
            "matches": pd.DataFrame(equals),
        }

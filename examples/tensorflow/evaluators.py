import pandas as pd
from mouffet.evaluation import Evaluator
from mouffet import common_utils
from sklearn.metrics import classification_report


class TFBasicEvaluator(Evaluator):
    def evaluate(self, data, options, infos):
        # * Just return the data provided by tensorflow
        common_utils.print_warning("Accuracy: {}".format(round(data["accuracy"], 3)))
        return {"stats": pd.DataFrame([data]), "matches": pd.DataFrame()}


class TFCustomEvaluator(Evaluator):
    def evaluate(self, data, options, infos):
        preds, labels, meta = data
        # * Get class with the best prediction score
        top_class = preds.to_numpy().argmax(axis=1)
        # * Get label names
        label_names = [meta.features["label"].int2str(int(x)) for x in preds.columns]
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

from multiprocessing.sharedctypes import Value
import pandas as pd
from mouffet.evaluation import Evaluator
from mouffet import common_utils
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt


class CustomEvaluator(Evaluator):
    def get_label_names(self, columns, metadata):
        res = []
        for x in columns:
            if x == -1:
                res.append("Unsure")
            else:
                res.append(metadata.features["label"].int2str(int(x)))
        return res

    def plot_confusion_matrix(self, data, options, infos):
        cm = confusion_matrix(data["labels"], data["predictions"])
        cm_plot = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=data["label_names"]
        )
        cm_plot.plot()
        plt.title(
            "Confusion matrix for model '{}'\nwith threshold: {}".format(
                infos["model"], options["threshold"]
            ),
            fontweight="bold",
            fontsize=12,
        )
        plt.xlabel("Predicted class", fontweight="bold")
        plt.ylabel("True class", fontweight="bold")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # cm_plot.ax_.set_xlabel(fontsize=12)
        # cm_plot.ax_.set_ylabel(fontsize=12)
        return cm_plot

    def evaluate(self, data, options, infos):
        res = {}
        preds, labels, meta = data
        # * Get class with the best prediction score
        thresh = options.get("threshold", -1)
        if thresh > 1:
            raise ValueError(
                "The option 'threshold' should only take values between 0 and 1 or -1"
            )
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

        res["stats"] = pd.DataFrame([cr])
        res["matches"] = pd.DataFrame(equals)

        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={
                    "labels": labels,
                    "predictions": top_class,
                    "label_names": label_names,
                },
                options=options,
                infos=infos,
            )

        return res

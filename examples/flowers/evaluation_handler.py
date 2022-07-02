import time

import pandas as pd
from mouffet.evaluation import EvaluationHandler

from plotnine import (
    aes,
    element_text,
    geom_point,
    ggplot,
    theme,
    theme_classic,
    xlab,
    ylab,
    scale_shape_discrete,
    scale_color_discrete,
)

import ast


class FlowersEvaluationHandler(EvaluationHandler):
    def get_evaluation_data(self, evaluator, database, model_opts, evaluator_opts):
        """This function returns the data that will be passed to the evaluators. All
        the required data should be returned there
        """
        # * Get raw data
        raw_data = self.data_handler.load_dataset("test", database, {})
        # * Get the predictions (will call predict_database)
        model_opts.opts["shuffle_data"] = False
        model_opts.opts["augment_data"] = False
        preds = self.get_predictions(model_opts, database)
        # * Get the labels
        labels = [y.numpy() for _, y in raw_data["data"]]
        # * Our custom evaluator will require: predictions, labels and metadata to get
        # * label names
        data = (preds, labels, raw_data["metadata"])

        return data

    def predict_database(self, model, database, db_type="test"):
        """This function calls a model to classify the database. The data to be classified
        is usually loaded there. This is because predictions can be saved to avoid the reclassification.
        This avoids loading the data for nothing.
        This function also logs general information about the classification
        """
        infos = {}
        ds = self.data_handler.load_dataset(
            db_type,
            database,
            {},
            prepare_opts=model.opts,
            prepare=True,
        )
        data = ds.data["data"]
        infos["n_images"] = len(data)
        start = time.time()
        preds = pd.DataFrame(model.predict(data))
        end = time.time()
        infos["global_duration"] = round(end - start, 2)
        infos["average_time_per_image"] = round(
            infos["global_duration"] / infos["n_images"], 2
        )
        infos["database"] = database.name

        return preds, infos

    def plot_accuracy_f1(self, res):
        stats = res["stats"]
        cust_stats = stats.loc[stats.evaluator == "custom"]
        if not cust_stats.empty:
            f1_scores = cust_stats
            f1_scores["f1"] = cust_stats["macro avg"].apply(lambda x: x.get("f1-score"))
            f1_scores["threshold"] = (
                cust_stats["evaluator_opts"]
                .apply(lambda x: ast.literal_eval(x).get("threshold"))
                .astype("category")
            )

        threshold_labels = [
            str(x) for x in f1_scores["threshold"].cat.categories.values
        ]
        threshold_labels[threshold_labels == -1] = "Max value"

        plt = (
            ggplot(
                data=f1_scores,
                mapping=aes(
                    x="accuracy",  # "factor(species, ordered=False)",
                    y="f1",
                    color="threshold",
                ),
            )
            + geom_point(
                aes(shape="model"),
                stat="identity",
                show_legend=True,
            )
            + xlab("Accuracy")
            + ylab("F1-Score")
            + theme_classic()
            + theme(
                axis_text_x=element_text(
                    angle=45,
                ),
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                text=element_text(size=12, weight="bold"),
            )
            + scale_shape_discrete(
                labels=[
                    "aug_rot-0.2_flip-H",
                    "aug_rot-0.2_flip-HV",
                    "aug_rot-0.3_flip-H",
                    "aug_rot-0.2_flip-HV",
                    "no_aug",
                ]
            )
            + scale_color_discrete(labels=threshold_labels)
        )
        return [plt]

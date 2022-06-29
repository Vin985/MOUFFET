import time

import pandas as pd
from mouffet.evaluation import EVALUATORS, EvaluationHandler

from data import FlowersDataHandler
from evaluators import CustomEvaluator
from plotnine import (
    aes,
    element_text,
    geom_point,
    ggplot,
    theme,
    theme_classic,
    xlab,
    ylab,
)

import ast

EVALUATORS.register_evaluator("custom", CustomEvaluator)


class FlowersEvaluationHandler(EvaluationHandler):
    def get_evaluation_data(self, evaluator, database, model_opts, evaluator_opts):
        """This function returns the data that will be passed to the evaluators. All
        the required data should be returned there
        """
        # * Get raw data
        raw_data = self.data_handler.load_dataset("test", database, {})
        # if evaluator_opts.get("use_raw_data", False):
        #     # * We evaluate the model directly from built-in functions
        #     # * Note: Evaluation is usually done in evaluators. However, evaluators
        #     # * do not have access to models and thus this is performed here
        #     model_opts.opts["augment_data"] = False
        #     model_opts.opts["shuffle_data"] = False
        #     model_opts.opts["inference"] = True
        #     model = self.load_model(model_opts)
        #     data = self.data_handler.prepare_dataset(raw_data, model_opts)
        #     data = model.model.evaluate(data["data"], return_dict=True)
        # else:
        # * We do everything manully
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

    def plot_test(self, res):
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
        )
        return [plt]


if __name__ == "__main__":
    evaluator = FlowersEvaluationHandler(
        opts_path="config/flowers/evaluation_config.yaml",
        dh_class=FlowersDataHandler,
    )

    res = evaluator.evaluate()

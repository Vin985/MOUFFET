import time

import pandas as pd
from mouffet.evaluation import EVALUATORS, EvaluationHandler

from data import TFExampleDataHandler
from evaluators import TFBasicEvaluator, TFCustomEvaluator

EVALUATORS.register_evaluator("tf", TFBasicEvaluator)
EVALUATORS.register_evaluator("custom", TFCustomEvaluator)


class TFExampleEvaluationHandler(EvaluationHandler):
    def get_evaluation_data(self, evaluator, database, model_opts, evaluator_opts):
        """This function returns the data that will be passed to the evaluators. All
        the required data should be returned there
        """
        # * Get raw data
        raw_data = self.data_handler.load_dataset("test", database, {})
        if evaluator_opts.get("use_raw_data", False):
            # * We evaluate the model directly from built-in functions
            # * Note: Evaluation is usually done in evaluators, however in mouffet evaluator
            # * do not have access to models and thus this is performed here
            model_opts.opts["augment_data"] = False
            model_opts.opts["shuffle_data"] = False
            model_opts.opts["inference"] = True
            model = self.load_model(model_opts)
            data = self.data_handler.prepare_dataset(raw_data, model_opts)
            data = model.model.evaluate(data["data"], return_dict=True)
        else:
            # * We do everything manully
            # * Get the predictions (will call predict_database)
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
        # data = model.prepare_data(ds)
        preds = pd.DataFrame(model.predict(data))
        end = time.time()
        infos["global_duration"] = round(end - start, 2)
        infos["average_time_per_image"] = round(
            infos["global_duration"] / infos["n_images"], 2
        )
        infos["database"] = database.name

        return preds, infos


evaluator = TFExampleEvaluationHandler(
    opts_path="examples/tensorflow/config/evaluation_config.yaml",
    dh_class=TFExampleDataHandler,
)

res = evaluator.evaluate()

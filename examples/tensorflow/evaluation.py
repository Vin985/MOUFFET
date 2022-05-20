import time

from mouffet.evaluation import EvaluationHandler, EVALUATORS, Evaluator

from data_handler import TFExampleDataHandler
import pandas as pd


class TFBasicEvaluator(Evaluator):
    def evaluate(self, data, options, infos):
        return {"stats": pd.DataFrame([data]), "matches": pd.DataFrame()}


EVALUATORS.register_evaluator("tf", TFBasicEvaluator)


class TFExampleEvaluationHandler(EvaluationHandler):
    def get_evaluation_data(self, evaluator, database, model_opts, evaluator_opts):
        if evaluator_opts.get("use_raw_data", False):
            data = self.data_handler.load_dataset(
                database,
                "test",
            )
            model_opts.opts["augment_data"] = False
            model_opts.opts["shuffle_data"] = False
            model_opts.opts["inference"] = True
            model = self.load_model(model_opts)
            data = model.prepare_data(data)
            data = model.model.evaluate(data, return_dict=True)
        else:
            data = self.get_predictions(model_opts, database)
        return data

    def classify_database(self, model, database, db_type="test"):
        infos = {}
        ds = self.data_handler.load_dataset(
            database,
            db_type,
        )
        infos["n_images"] = len(ds["data"])
        start = time.time()
        preds = pd.DataFrame(model.predict(model.prepare_data(ds)))
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

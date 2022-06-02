from mouffet.evaluation import EVALUATORS

from data_handler import TFExampleDataHandler
from evaluation_handler import TFExampleEvaluationHandler
from evaluators import TFBasicEvaluator, TFCustomEvaluator

EVALUATORS.register_evaluator("tf", TFBasicEvaluator)
EVALUATORS.register_evaluator("custom", TFCustomEvaluator)

evaluator = TFExampleEvaluationHandler(
    opts_path="examples/tensorflow/config/evaluation_config.yaml",
    dh_class=TFExampleDataHandler,
)

res = evaluator.evaluate()

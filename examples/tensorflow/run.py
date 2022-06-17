from mouffet.evaluation import EVALUATORS
from mouffet.runs import RunArgumentParser, launch_runs
from mouffet.training.training_handler import TrainingHandler

from data import TFExampleDataHandler
from evaluation_handler import TFExampleEvaluationHandler
from evaluators import TFBasicEvaluator, TFCustomEvaluator

EVALUATORS.register_evaluator("tf", TFBasicEvaluator)
EVALUATORS.register_evaluator("custom", TFCustomEvaluator)


parser = RunArgumentParser()
args = parser.parse_args()

launch_runs(
    args,
    handler_classes={
        "training": TrainingHandler,
        "data": TFExampleDataHandler,
        "evaluation": TFExampleEvaluationHandler,
    },
)

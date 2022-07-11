from mouffet.evaluation import EVALUATORS
from mouffet.runs import RunHandler
from mouffet.training.training_handler import TrainingHandler

from data import FlowersDataHandler
from evaluation import FlowersEvaluationHandler
from evaluators import CustomEvaluator

EVALUATORS.register_evaluator("custom", CustomEvaluator)

run_handler = RunHandler(
    handler_classes={
        "training": TrainingHandler,
        "data": FlowersDataHandler,
        "evaluation": FlowersEvaluationHandler,
    },
    default_args={
        "run_dir": "config",
    },
)

run_handler.launch_runs()

from mouffet.evaluation import EVALUATORS

from data import FlowersDataHandler
from evaluators import CustomEvaluator

from evaluation_handler import FlowersEvaluationHandler

EVALUATORS.register_evaluator("custom", CustomEvaluator)

if __name__ == "__main__":
    evaluator = FlowersEvaluationHandler(
        opts_path="config/flowers/evaluation_config.yaml",
        dh_class=FlowersDataHandler,
    )

    res = evaluator.evaluate()

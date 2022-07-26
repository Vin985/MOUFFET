from .evaluator import Evaluator


class _Evaluators:
    def __init__(self):
        self._evaluators = {}

    def register_evaluator(self, evaluator):
        error = False
        if isinstance(evaluator, type):
            if issubclass(evaluator, Evaluator):
                evaluator = evaluator()
            else:
                error = True
        elif not issubclass(evaluator.__class__, Evaluator):
            error = True
        if error:
            raise ValueError(
                "evaluator should be either a class or an instance"
                + " of a subclass of the mouffet.evaluation.Evaluator class"
            )
        self._evaluators[evaluator.NAME] = evaluator

    def register_evaluators(self, evaluators):
        # if not isinstance(evaluators, dict):
        #     raise AttributeError("'evaluators' should be a dict")
        for evaluator in evaluators:
            self.register_evaluator(evaluator)

    def __getitem__(self, name):
        if name is None:
            raise ValueError(
                "An evaluator name should be provided using the 'type' option in the"
                + " evaluation configuration file"
            )
        if not name in self._evaluators:
            raise ValueError(
                (
                    "Evaluator {} is not registered. Please add it"
                    + " using the register_evaluator() method"
                ).format(name)
            )
        return self._evaluators[name]


EVALUATORS = _Evaluators()


from .evaluation_handler import EvaluationHandler

import os
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

from data_handler import TFExampleDataHandler


evaluator = SongDetectorEvaluationHandler(
    opts_path="examples/tensorflow/config/evaluation_config.yaml",
    dh_class=TFExampleDataHandler,
)

res = evaluator.evaluate()

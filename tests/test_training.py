from mouffet.training import TrainingHandler

from flowers.data import FlowersDataHandler
import os


def test_nscenarios():
    print(os.getcwd())
    trainer = TrainingHandler(
        opts_path="tests/config/training_config.yaml",
        dh_class=FlowersDataHandler,
    )
    assert len(trainer.scenarios) == 5


def test_repeat():
    print(os.getcwd())
    trainer = TrainingHandler(
        opts_path="tests/config/training/training_repeat.yaml",
        dh_class=FlowersDataHandler,
    )
    assert len(trainer.scenarios) == 3

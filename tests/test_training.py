from mouffet.training import TrainingHandler

from flowers.data import FlowersDataHandler
import os


def test_nscenarios():
    trainer = TrainingHandler(
        opts_path="tests/config/training_config.yaml",
        dh_class=FlowersDataHandler,
    )
    assert len(trainer.scenarios) == 5


def test_repeat():
    trainer = TrainingHandler(
        opts_path="tests/config/training/training_repeat.yaml",
        dh_class=FlowersDataHandler,
    )
    assert len(trainer.scenarios) == 3


def test_already_trained():
    trainer = TrainingHandler(
        opts_path="tests/config/training/training_already_trained.yaml",
        dh_class=FlowersDataHandler,
    )
    assert trainer.train() == [2]

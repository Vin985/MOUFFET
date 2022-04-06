from mouffet.training import TrainingHandler

from data_handler import TFExampleDataHandler

trainer = TrainingHandler(
    opts_path="examples/tensorflow/config/training_config.yaml",
    dh_class=TFExampleDataHandler,
)
trainer.train()

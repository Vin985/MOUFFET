from mouffet.training import TrainingHandler

from data import FlowersDataHandler

if __name__ == "__main__":
    trainer = TrainingHandler(
        opts_path="config/flowers/training_config.yaml",
        dh_class=FlowersDataHandler,
    )
    trainer.train()

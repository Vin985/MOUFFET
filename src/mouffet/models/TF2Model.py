from abc import abstractmethod
from pathlib import Path

import mouffet.utils.common as common_utils
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .dlmodel import DLModel


class TF2Model(DLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.summary_writer = {}
        self.metrics = {}

    @property
    def n_parameters(self):
        if self.model:
            res = {}
            res["trainableParams"] = np.sum(
                [
                    np.prod(v.get_shape())
                    for v in self.model.trainable_weights  # pylint: disable=no-member
                ]
            )
            res["nonTrainableParams"] = np.sum(
                [
                    np.prod(v.get_shape())
                    for v in self.model.non_trainable_weights  # pylint: disable=no-member
                ]
            )
            res["totalParams"] = res["trainableParams"] + res["nonTrainableParams"]
            return str(res)
        else:
            return super().n_parameters

    @tf.function
    def train_step(self, data, labels):
        self.basic_step(data, labels, self.STEP_TRAINING)

    @tf.function
    def validation_step(self, data, labels):
        self.basic_step(data, labels, self.STEP_VALIDATION)

    def basic_step(self, data, labels, step_type):
        training = step_type == self.STEP_TRAINING
        if training:
            with tf.GradientTape() as tape:
                predictions = self.model(data, training=True)
                loss = self.tf_loss(labels, predictions)
            gradients = tape.gradient(
                loss, self.model.trainable_variables  # pylint: disable=no-member
            )  # pylint: disable=no-member
            self.optimizer.apply_gradients(
                zip(
                    gradients, self.model.trainable_variables
                )  # pylint: disable=no-member
            )
        else:
            predictions = self.model(data, training=False)
            loss = self.tf_loss(labels, predictions)

        self.metrics[step_type + "_loss"](loss)
        self.metrics[step_type + "_accuracy"](labels, predictions)

    @staticmethod
    @abstractmethod
    def tf_loss(y_true, y_pred):
        """This is the loss function.

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def init_metrics(self):
        """Inits the metrics used during model evaluation. Fills the metrics
        attribute which is a dict that should contain the following keys:

        - train_loss
        - train_accuracy
        - validation_loss
        - validation accuracy
        """
        raise NotImplementedError()

    @abstractmethod
    def init_samplers(self, training_data, validation_data):
        raise NotImplementedError()

    @abstractmethod
    def init_optimizer(self, learning_rate):
        raise NotImplementedError()

    def init_model(self):
        self.model = self.create_model()
        if "weights_opts" in self.opts or self.opts.get("inference", False):
            self.load_weights()

    def init_training(self):
        """This is a function called at the beginning of the training step. In
        this function you should initialize your train and validation samplers,
        as well as the optimizer, loss and accuracy functions for training and
        validation.

        """
        self.opts.add_option("training", True)

        self.init_model()

        self.init_metrics()

    def run_epoch(
        self,
        epoch,
        training_sampler,
        validation_sampler,
        epoch_save_step=None,
    ):
        # * Reset the metrics at the start of the next epoch
        self.reset_states()

        self.run_step("train", epoch, training_sampler)
        self.run_step("validation", epoch, validation_sampler)

        template = (
            "Epoch {}, Loss: {}, Accuracy: {},"
            " Validation Loss: {}, Validation Accuracy: {}"
        )
        print(
            template.format(
                epoch,
                self.metrics["train_loss"].result(),
                self.metrics["train_accuracy"].result() * 100,
                self.metrics["validation_loss"].result(),
                self.metrics["validation_accuracy"].result() * 100,
            )
        )

        if epoch_save_step is not None and epoch % epoch_save_step == 0:
            self.save_model(self.opts.get_intermediate_path(epoch))

    def train(self, training_data, validation_data):

        self.init_training()

        print("Training model", self.opts.model_id)

        early_stopping = self.opts.get("early_stopping", {})
        stop = False
        if early_stopping:
            patience = early_stopping.get("patience", 3)
            count = 1

        training_stats = {"crossed": False}

        training_sampler, validation_sampler = self.init_samplers(
            training_data, validation_data
        )

        epoch_save_step = self.opts.get("epoch_save_step", None)

        epoch_batches = []

        # * Create logging writers
        self.create_writers()

        epoch_batches = self.get_epoch_batches()

        for batch in epoch_batches:
            lr = batch["learning_rate"]

            common_utils.print_info(
                (
                    "Starting new batch of epochs from epoch number {}, with learning rate {} for {} iterations"
                ).format(batch["start"], lr, batch["length"])
            )

            if batch.get("fine_tuning", False):
                print("Doing fine_tuning")
                self.set_fine_tuning()
                self.model.summary()  # pylint: disable=no-member

            self.init_optimizer(learning_rate=lr)
            for epoch in range(batch["start"], batch["end"] + 1):
                print("Running epoch ", epoch)
                self.run_epoch(
                    epoch,
                    training_sampler,
                    validation_sampler,
                    epoch_save_step,
                )
                train_loss = self.metrics["train_loss"].result()
                val_loss = self.metrics["validation_loss"].result()

                diff = train_loss - val_loss

                if diff <= 0:
                    if not training_stats["crossed"]:
                        training_stats["crossed"] = True
                        training_stats["crossed_at"] = epoch
                        self.save_model(self.opts.get_intermediate_path(epoch))

                    if early_stopping:
                        if count < patience:
                            count += 1
                        else:
                            stop = True
                            break
                else:
                    count = 0

            if stop:
                break
                # training_stats["train_loss"] = train_loss
                # training_stats["val_loss"] = val_loss

        self.save_model()
        return training_stats

    def create_writers(self):
        log_dir = Path(self.opts.logs["log_dir"]) / (
            self.opts.model_id + "_v" + str(self.opts.save_version)
        )

        # TODO: externalize logging directory
        train_log_dir = log_dir / "train"
        validation_log_dir = log_dir / "validation"
        self.summary_writer["train"] = tf.summary.create_file_writer(str(train_log_dir))
        self.summary_writer["validation"] = tf.summary.create_file_writer(
            str(validation_log_dir)
        )

    def reset_states(self):
        for x in ["train", "validation"]:
            self.metrics[x + "_loss"].reset_states()
            self.metrics[x + "_accuracy"].reset_states()

    def run_step(self, step_type, step, sampler):
        for data, labels in tqdm(sampler, ncols=50):
            getattr(self, step_type + "_step")(data, labels)

        with self.summary_writer[step_type].as_default():
            tf.summary.scalar(
                "loss", self.metrics[step_type + "_loss"].result(), step=step
            )
            tf.summary.scalar(
                "accuracy", self.metrics[step_type + "_accuracy"].result(), step=step
            )

    def save_weights(self, path=None):
        if not path:
            path = str(self.opts.results_save_dir / self.opts.model_id)
        self.model.save_weights(path)  # pylint: disable=no-member

    def load_weights(self):
        self.model.load_weights(  # pylint: disable=no-member
            self.opts.get_weights_path()
        )

    @abstractmethod
    def predict(self, x):
        """This function calls the model to have a predictions

        Args:
            x (data): The input data to be classified

            NotImplementedError: No basic implementation is provided and it should therefore be
            provided in child classes
        """
        raise NotImplementedError()

from abc import abstractmethod
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

import mouffet.utils.common as common_utils

from .dlmodel import DLModel


class TF2Model(DLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.summary_writer = {}
        self.metrics = {}

    @tf.function
    def train_step(self, data, labels):
        self.basic_step(data, labels, self.STEP_TRAINING)

    @tf.function
    def validation_step(self, data, labels):
        self.basic_step(data, labels, self.STEP_VALIDATION)

    def basic_step(self, data, labels, step_type):
        training = step_type == self.STEP_VALIDATION
        if not training:
            with tf.GradientTape() as tape:
                predictions = self.model(data, training=True)
                loss = self.tf_loss(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
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
    def init_samplers(self):
        raise NotImplementedError()

    @abstractmethod
    def init_optimizer(self, learning_rate):
        raise NotImplementedError()

    @abstractmethod
    def init_model(self):
        raise NotImplementedError()

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
        training_data,
        validation_data,
        training_sampler,
        validation_sampler,
        epoch_save_step=None,
    ):
        # * Reset the metrics at the start of the next epoch
        self.reset_states()

        self.run_step("train", training_data, epoch, training_sampler)
        self.run_step("validation", validation_data, epoch, validation_sampler)

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

        training_sampler, validation_sampler = self.init_samplers()

        epoch_save_step = self.opts.get("epoch_save_step", None)

        # * Create logging writers
        self.create_writers()

        n_epochs = self.opts["n_epochs"]
        learning_rates = self.opts["learning_rate"]
        from_epoch = self.opts.get("from_epoch", 0)

        # * Convert number of epochs to list for iteration
        if not isinstance(n_epochs, list):
            n_epochs = [n_epochs]

        # * Convert learning rates to list
        if not isinstance(learning_rates, list):
            learning_rates = [learning_rates] * len(n_epochs)
        elif len(learning_rates) < len(n_epochs):
            # * Reuse the last value for all missing epochs
            common_utils.print_warning(
                (
                    "Smaller number of learning rates then epochs found."
                    + " Reusing the last value for the learning rate for all missing epochs."
                )
            )
            learning_rates = learning_rates + learning_rates[-1:] * (
                len(n_epochs) - len(learning_rates)
            )

        batch_starts = [1]

        start_idx = -1
        # * Get epoch batch index if it we resume training from a specific epoch
        if from_epoch:
            epoch_count = 0
            for i, batch_len in enumerate(n_epochs):
                epoch_count += batch_len
                if from_epoch <= epoch_count and start_idx < 0:
                    start_idx = i
                if len(batch_starts) < len(n_epochs):
                    batch_starts.append(batch_starts[i] + batch_len)

        start_idx = start_idx if start_idx >= 0 else 0

        # * Iterate over all batches
        current_batch = start_idx
        for batch in n_epochs[current_batch:]:
            # * Get learning rate for the current batch
            lr = learning_rates[current_batch]
            # * Get the real start of the batch if from_epoch is specified
            batch_start = max(batch_starts[current_batch], from_epoch)
            # * Get the length of the batch for the loop
            batch_end = batch_starts[current_batch] + batch
            current_batch += 1

            common_utils.print_info(
                (
                    "Starting new batch of epochs from epoch number {}, with learning rate {} for {} iterations"
                ).format(batch_start, lr, batch_end - batch_start)
            )

            self.init_optimizer(learning_rate=lr)
            for epoch in range(batch_start, batch_end):
                self.run_epoch(
                    epoch,
                    training_data,
                    validation_data,
                    training_sampler,
                    validation_sampler,
                    epoch_save_step,
                )

        self.save_model()

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

    def run_step(self, step_type, data, step, sampler):
        for data, labels in tqdm(
            sampler(self.get_raw_data(data), self.get_ground_truth(data))
        ):

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
        self.model.save_weights(path)

    def load_weights(self):
        self.model.load_weights(self.opts.get_weights_path())

    @abstractmethod
    def predict(self, x):
        """This function calls the model to have a predictions

        Args:
            x (data): The input data to be classified

            NotImplementedError: No basic implementation is provided and it should therefore be
            provided in child classes
        """
        raise NotImplementedError()

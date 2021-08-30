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

    def get_epoch_batches(self):
        n_epochs = self.opts["n_epochs"]
        learning_rates = self.opts["learning_rate"]
        from_epoch = self.opts.get("from_epoch", 0)

        # * Convert number of epochs to list for iteration
        if not isinstance(n_epochs, list):
            n_epochs = [n_epochs]

        if not isinstance(learning_rates, list):
            learning_rates = [learning_rates] * len(n_epochs)

        epoch_count = 0
        from_epoch_met = False
        next_batch_start = 1
        epoch_batches = []
        for i, batch_len in enumerate(n_epochs):

            # * From epoch greater than total count: skip batch
            epoch_count += batch_len
            if from_epoch > epoch_count:
                continue

            batch = {}

            if from_epoch and not from_epoch_met:
                batch["start"] = from_epoch
                from_epoch_met = True
            else:
                batch["start"] = next_batch_start

            next_batch_start += batch_len

            batch["end"] = epoch_count
            batch["length"] = batch["end"] - batch["start"] + 1

            if i >= len(learning_rates):
                batch["learning_rate"] = learning_rates[-1]
            else:
                batch["learning_rate"] = learning_rates[i]

            epoch_batches.append(batch)

        if self.opts.get("transfer_learning", False):
            print("doing tl")
            fine_tuning = self.opts.get("fine_tuning", {})
            print(fine_tuning)
            if fine_tuning:
                batch = {}
                batch_len = fine_tuning.get("n_epochs", 0)
                lr = fine_tuning.get("learning_rate", 0)
                if not batch_len:
                    common_utils.print_warning(
                        "n_epochs option is not specified for fine tuning. Cannot proceed."
                    )
                elif not lr:
                    common_utils.print_warning(
                        "learning_rate option is not specified for fine tuning. Cannot proceed."
                    )
                else:
                    batch["start"] = epoch_count + 1
                    epoch_count += batch_len
                    batch["end"] = epoch_count
                    batch["learning_rate"] = lr
                    batch["length"] = batch["end"] - batch["start"] + 1
                    batch["fine_tuning"] = True
                    epoch_batches.append(batch)
        return epoch_batches

    def train(self, training_data, validation_data):

        self.init_training()

        print("Training model", self.opts.model_id)

        training_sampler, validation_sampler = self.init_samplers()

        epoch_save_step = self.opts.get("epoch_save_step", None)

        epoch_batches = []

        # * Create logging writers
        self.create_writers()

        epoch_batches = self.get_epoch_batches()

        print(epoch_batches)

        for batch in epoch_batches:
            lr = batch["learning_rate"]

            common_utils.print_info(
                (
                    "Starting new batch of epochs from epoch number {}, with learning rate {} for {} iterations"
                ).format(batch["start"], lr, batch["length"])
            )

            if batch.get("fine_tuning", False):
                print("Doing fine_tuning")

            self.init_optimizer(learning_rate=lr)
            for epoch in range(batch["start"], batch["end"] + 1):
                print("Running epoch ", epoch)
        #         self.run_epoch(
        #             epoch,
        #             training_data,
        #             validation_data,
        #             training_sampler,
        #             validation_sampler,
        #             epoch_save_step,
        #         )

        # self.save_model()

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

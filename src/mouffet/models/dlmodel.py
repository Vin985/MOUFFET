import logging
from abc import abstractmethod

from ..utils import common_utils
from .model import Model


class DLModel(Model):
    """Subclass of mouffet.models.model.Model for deep learning models"""

    NAME = "DLMODEL"
    NETWORK_OPTION_FILENAME = "network_opts.yaml"

    def init_model(self):
        self.model = self.create_model()
        if "weights_opts" in self.opts or self.opts.get("inference", False):
            self.load_weights()

    @abstractmethod
    def create_model(self):
        """Basic implementation of model creation for deep learning models. By default only creates
        the network

        Returns:
            Object: The network created by the "create_net" method
        """
        return 0

    @abstractmethod
    def train(self, training_data, validation_data):
        """Function call by the training handler to train the model

        Args:
            training_data (Object): The training data containing raw data and ground truth
            validation_data (Object): The validation data containing raw data and ground truth

        Raises:
            NotImplementedError: Class must be inherited
        """
        raise NotImplementedError("train function not implemented for this class")

    @abstractmethod
    def save_weights(self, path=None):
        """Save the weights of the model

        Args:
            path (str or pathlib.Path, optional): The path where the weigths should be saved.
            Defaults to None.

        Raises:
            NotImplementedError: Class must be inherited
        """
        raise NotImplementedError(
            "save_weights function not implemented for this class"
        )

    @abstractmethod
    def load_weights(self):
        """Load the weights of the model

        Raises:
            NotImplementedError: Class must be inherited
        """
        raise NotImplementedError(
            "load_weights function not implemented for this class"
        )

    def save_params(self):
        """Save network options"""
        self.save_options(self.NETWORK_OPTION_FILENAME, self.opts.opts)

    def save_model(self, path=None):
        """Default implementation for deep learning model saving.
        Calls save_params to save the network options and save_weigths to save the weights of
        the model.

        Args:
            path ([type], optional): [description]. Defaults to None.
        """
        self.save_params()
        self.save_weights(path)

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
            if batch_len:
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
            logging.debug("Performing transfer learning, retrieving additional batches")
            fine_tuning = self.opts.get("fine_tuning", {})
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

    def set_fine_tuning(self):
        pass

from abc import abstractmethod

from .model import Model

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class DLModel(Model):
    """Subclass of mouffet.models.model.Model for deep learning models
    """

    NAME = "DLMODEL"

    @abstractmethod
    def create_model(self):
        """Basic implementation of model creation for deep learning models. By default only creates
        the network

        Returns:
            Object: The network created by the "create_net" method
        """
        return 0

    # @abstractmethod
    # def create_net(self):
    #     """Creates the network for the deep learning model

    #     Returns:
    #         Object: The deep learning network
    #     """
    #     return 0

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
            "save_weights function not implemented for this class"
        )

    def save_params(self):
        """Save network options
        """
        self.save_options("network_opts.yaml", self.opts.opts)

    def save_model(self, path=None):
        """Default implementation for deep learning model saving.
        Calls save_params to save the network options and save_weigths to save the weights of
        the model.

        Args:
            path ([type], optional): [description]. Defaults to None.
        """
        self.save_params()
        self.save_weights(path)

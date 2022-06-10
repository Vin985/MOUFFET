from abc import ABC, abstractmethod

import yaml

from ..utils import file as file_utils


class Model(ABC):
    """Base abstract class for defining models. Must be subclassed.

    Attributes:
        NAME: Name of the model class. Will be used to identify and save classes
        STEP_TRAINING: The name of the training step. Will be used in configuration file
        STEP_VALIDATION: The name of the validation step. Will be used in configuration file
    """

    NAME = "MODEL"

    STEP_TRAINING = "train"
    STEP_VALIDATION = "validation"

    def __init__(self, opts=None):
        """Create the layers of the neural network, with the same options we used in training"""
        self.model = None
        self._opts = None
        if opts:
            self.opts = opts

    def check_options(self):
        return True

    @property
    def n_parameters(self):
        return -1

    @property
    def opts(self):
        """Property that contains the options related to the model as read in the configuration file"""
        return self._opts

    @opts.setter
    def opts(self, opts):
        """When setting options the name of the model as described by the NAME attribute is set in
        the options and the method create_model() is called to initialize the model

        Args:
            opts (dict): Model options
        """
        self._opts = opts
        if not opts["name"]:
            self._opts.opts["name"] = self.NAME

    @abstractmethod
    def create_model(self):
        """Abstract method where model creation / network initialization should take place

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError(
            "create_model function not implemented for this class"
        )

    @abstractmethod
    def predict(self, x):
        """Predict data using the model

        Args:
            x (data): the data on which we want to makae a prediction. This function should make only
            one prediction and be called numerous times if needed

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("predict function not implemented for this class")

    @abstractmethod
    def classify(self, data, sampler):
        """Perform classification on several data items using the sampler. This function should call
        predict.

        Args:
            data (Object): The data to classify
            sampler (Object): An optional object that performs subsampling of the data and returns
            objects that can be passed to the predict() method.

        Returns:
            Object: The predicted data
        """
        return None

    # @abstractmethod
    # def get_ground_truth(self, data):
    #     """Get ground truth data from the dataset

    #     Args:
    #         data (Object): A dataset defined by the data structure of the data handler

    #     Returns:
    #         Object: The ground truth data
    #     """
    #     return data

    # @abstractmethod
    # def get_raw_data(self, data):
    #     """Get raw data from the dataset

    #     Args:
    #         data (Object): A dataset defined by the data structure of the data handler

    #     Returns:
    #         Object: The raw truth data
    #     """
    #     return data

    @abstractmethod
    def save_model(self, path=None):
        """Save the model to disk

        Args:
            path (Object, optional): Path where the model should be saved. Defaults to None.

        Raises:
            NotImplementedError: Should be inherited by the final model
        """
        raise NotImplementedError("save_model function not implemented for this class")

    def prepare_data(self, data):
        """Prepare the data before training. Allows to perform last minute changes to the data just before
        training.

        Args:
            data (Object): The data to be prepared

        Returns:
            Object: The modified data
        """
        return data

    def save_options(self, file_name, options):
        """Save the options related to the model for logging purposes as a yaml file. By default, uses the
        "results_save_dir" property from the ModelOptions class associated to the model.
        By default, this will be the following combination:
        model_dir/model_id/version
        where model_dir and model_id can be found in the model configuration file and version is
        calculated automatically.

        Args:
            file_name (string or pathlib.Path): The name of the file to be saved
            options (Object): The options to save that can be transcribed as a yaml file
        """
        file_utils.ensure_path_exists(self.opts.results_save_dir)
        with open(self.opts.results_save_dir / file_name, "w") as f:
            yaml.dump(options, f, default_flow_style=False)

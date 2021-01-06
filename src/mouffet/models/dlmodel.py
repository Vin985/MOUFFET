from abc import ABC, abstractmethod

import yaml

from ..utils import file as file_utils

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class DLModel(ABC):
    NAME = "DLMODEL"

    STEP_TRAINING = "train"
    STEP_VALIDATION = "validation"

    def __init__(self, opts=None):
        """Create the layers of the neural network, with the same options we used in training"""
        self.model = None
        self._results_dir = None
        self._opts = None
        if opts:
            self.opts = opts

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, opts):
        self._opts = opts
        self._opts.opts["name"] = self.NAME
        self.model = self.create_net()

    @abstractmethod
    def create_net(self):
        return 0

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("predict function not implemented for this class")

    @abstractmethod
    def train(self, training_data, validation_data):
        raise NotImplementedError("train function not implemented for this class")

    @abstractmethod
    def save_weights(self, path=None):
        raise NotImplementedError(
            "save_weights function not implemented for this class"
        )

    @abstractmethod
    def load_weights(self, path=None):
        raise NotImplementedError(
            "load_weights function not implemented for this class"
        )

    @abstractmethod
    def classify(self, data, sampler):
        return None

    @abstractmethod
    def get_ground_truth(self, data):
        return data

    @abstractmethod
    def get_raw_data(self, data):
        return data["spectrograms"]

    def prepare_data(self, data):
        return data

    def save_params(self):
        file_utils.ensure_path_exists(self.opts.results_save_dir)
        with open(self.opts.results_save_dir / "network_opts.yaml", "w") as f:
            yaml.dump(self.opts.opts, f, default_flow_style=False)

    def save_model(self, path=None):
        self.save_params()
        self.save_weights(path)

from abc import abstractmethod

import yaml
from .model import Model
from ..utils import file as file_utils

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class DLModel(Model):
    NAME = "DLMODEL"

    STEP_TRAINING = "train"
    STEP_VALIDATION = "validation"

    def create_model(self):
        return self.create_net()

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
    def load_weights(self, path=None, from_epoch=None):
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
        return data

    def prepare_data(self, data):
        return data

    def save_options(self, file_name, options):
        file_utils.ensure_path_exists(self.opts.results_save_dir)
        with open(self.opts.results_save_dir / file_name, "w") as f:
            yaml.dump(options, f, default_flow_style=False)

    def save_params(self):
        self.save_options("network_opts.yaml", self.opts.opts)

    def save_model(self, path=None):
        self.save_params()
        self.save_weights(path)

from abc import ABC, abstractmethod
from importlib import import_module

import pydoc

from ..data.data_handler import DataHandler
from ..options.model_options import ModelOptions
from ..utils import file as file_utils


class ModelHandler(ABC):
    def __init__(
        self, opts=None, opts_path="", dh=None, model=None, model_class=None, **kwargs,
    ):
        if not opts:
            if opts_path:
                opts = file_utils.load_config(opts_path)
            else:
                raise AttributeError(
                    "You should provide either an options dict via the opts attribute"
                    + "or the path to the config file via opts_path"
                )
        self.opts = opts
        self._model = None
        self.model_class = model_class
        self._data_handler = None
        if model:
            self.model = model
        if not dh:
            dh = self.create_data_handler(**kwargs)
        self.data_handler = dh

        self.scenarios = self.load_scenarios()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model and not model.opts:
            model.opts = self.opts
        self._model = model

    @property
    def data_handler(self):
        return self._data_handler

    @data_handler.setter
    def data_handler(self, dh):
        if dh and not dh.opts:
            dh.opts = self.opts
        self._data_handler = dh

    def create_data_handler(self, dh_class=None, split_funcs=None):
        data_opts_path = self.opts.get("data_config", "")
        if not data_opts_path:
            raise Exception("A path to the data configuration file must be provided")
        data_opts = file_utils.load_config(data_opts_path)
        if dh_class and issubclass(dh_class, DataHandler):
            dh = dh_class(data_opts, split_funcs=split_funcs)
        else:
            raise Exception("A subclass of DataHandler must be provided")
        return dh

    def get_model_instance(self, model_opts):
        if not isinstance(model_opts, ModelOptions):
            raise ValueError(
                "Argument 'model' should be an instance of dlbd.options.ModelOptions or a dict"
            )

        # pkg = import_module(model["package"])
        # cls = getattr(pkg, model["name"])
        cls = pydoc.locate(model_opts["class"])
        return cls(model_opts)

    def get_option(self, name, group, default=""):
        return group.get(name, self.opts.get(name, default))

    @abstractmethod
    def load_scenarios(self):
        return []


import pydoc
from abc import ABC, abstractmethod
from pathlib import Path

from ..data import DataHandler
from ..models import DLModel
from ..options import ModelOptions
from ..utils import common_utils, file_utils


class ModelHandler(ABC):

    DATA_HANDLER_CLASS = DataHandler
    NETWORK_OPTION_FILENAME = "network_opts.yaml"

    def __init__(
        self,
        opts=None,
        opts_path="",
        dh=None,
        dh_class=None,
        **kwargs,
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
        self._data_handler = None
        if dh_class:
            self.DATA_HANDLER_CLASS = dh_class
        if dh:
            self.data_handler = dh

        self.scenarios = self.load_scenarios()

    @property
    def data_handler(self):
        if not self._data_handler:
            self.data_handler = self.create_data_handler()
        return self._data_handler

    @data_handler.setter
    def data_handler(self, dh):
        if dh and not dh.opts:
            dh.opts = self.opts
        self._data_handler = dh

    def create_data_handler(self):
        data_opts_path = self.opts.get("data_config", "")
        if not data_opts_path:
            raise Exception("A path to the data configuration file must be provided")
        data_opts = file_utils.load_config(data_opts_path)
        if self.DATA_HANDLER_CLASS and issubclass(self.DATA_HANDLER_CLASS, DataHandler):
            dh = self.DATA_HANDLER_CLASS(data_opts)
        else:
            raise Exception("A subclass of DataHandler must be provided")
        return dh

    @classmethod
    def load_model(cls, model_opts):
        """Load a model from the options provided by model_opts. Note: the options
        saved during training will be loaded and overriden by the relevant options from
        model_opts (especially the paths). One exception is the "id" and "id_prefixes" options
        as the one from the old options will always be used to prevent any conflict if the user
        decides to define a new id for saving the evaluation results.

        Args:
            model_opts (mouffet.options.ModelOptions): The model options for the current scenario

        Returns:
            mouffet.model.DLModel: the loaded model
        """
        ignore_parent_path = model_opts.get("ignore_parent_path", False)
        version = model_opts.load_version

        # * If in transfer learning, load the full model if information is provided in the weights
        weight_opts = model_opts.get("weights_opts", {})
        model_dir = weight_opts.get("model_dir", model_opts.model_dir)
        model_id = weight_opts.get("name", model_opts.model_id)
        # model_name = weight_opts.get("name", model_opts.name)

        old_opts = file_utils.load_config(
            Path(model_dir) / model_id / str(version) / cls.NETWORK_OPTION_FILENAME,
            ignore_parent_path,
        )
        old_opts["data_config"] = model_opts.get("data_config", "")
        # * To load the model, we use the old opts updated by the scenario, except for the id
        opts = ModelOptions(old_opts)

        # * Rename ids in transfer learning, not when performing inference
        # except_keys = ["id", "id_prefixes"]
        except_keys = (
            ["suffix", "suffix_prepend"]
            if not model_opts.get("transfer_learning", False)
            else []
        )

        if model_opts.get("inference", False) and "weights_opts" in opts:
            # * Remove old weight_opts if in inference mode to avoid conflicts
            opts.opts.pop("weights_opts")

        common_utils.deep_dict_update(
            opts.opts, model_opts.opts, except_keys=except_keys
        )

        model = cls.get_model_instance(opts)
        model.init_model()
        return model

    @staticmethod
    def get_model_instance(model_opts):
        if not isinstance(model_opts, ModelOptions):
            raise ValueError(
                "Argument 'model' should be an instance of dlbd.options.ModelOptions or a dict"
            )
        model_class = model_opts["class"]
        if isinstance(model_class, str):
            cls = pydoc.locate(model_class)
        elif issubclass(model_class, DLModel):
            cls = model_class
        else:
            raise ValueError(
                "Option 'class' should either be a subclass of",
                "mouffet.models.dlmodel.DLModel or a string",
            )

        if not cls:
            raise ValueError("No class named {} was found".format(model_class))

        return cls(model_opts)

    def get_option(self, name, group, default=""):
        return group.get(name, self.opts.get(name, default))

    @abstractmethod
    def load_scenarios(self):
        return []

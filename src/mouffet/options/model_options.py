import re
import shutil

from copy import deepcopy

from .options import Options
from ..utils import common_utils


class ModelOptions(Options):

    DEFAULT_VALUES = {
        "id": "",
        "id_prefixes": {"default": ""},
        "intermediate_save_dir": "intermediate",
    }

    DICT_PATH_SEPARATOR = "--"

    def __init__(self, opts):
        super().__init__(opts)
        self._model_id = ""
        self._previous_version = None

    @property
    def results_dir_root(self):
        return self.model_dir / self.model_id

    @property
    def results_save_dir(self):
        return self.get_results_dir()

    @property
    def results_load_dir(self):
        return self.get_results_dir(save=False)

    def version(self, save=True):
        default = self.previous_version
        if save or not default:
            # * Add one if we are saving or if we are loading and the previous number is 0
            default += 1
        return self.opts.get("version", default)

    @property
    def previous_version(self):
        if self._previous_version is None:
            self._previous_version = self.get_last_version()
        return self._previous_version

    @property
    def load_version(self):
        return self.version(save=False)

    @property
    def save_version(self):
        return self.version(save=True)

    def get_results_dir(self, save=True):
        return self.results_dir_root / str(self.version(save))

    def get_weights_path(self, epoch=None, version=None, as_string=True):
        weight_opts = self.get("weights_opts", {})
        path = weight_opts.get("path", "")
        if path:
            return path

        name = weight_opts.get("name", "")
        if name and name != self.model_id:
            # * Load weights from another model
            # * For that, create a new model options with the new name
            tmp_opts = ModelOptions(deepcopy(self.opts))
            tmp_opts.model_id = name
            # * Update options from weight opts
            # TODO: allow redefining?
            if "model_dir" in weight_opts:
                tmp_opts.add_option("model_dir", weight_opts["model_dir"])
            path = tmp_opts.get_weights_path()
            return path
        from_epoch = weight_opts.get("from_epoch", 0)
        if from_epoch:
            if version is None:
                version = weight_opts.get("version", -1)
            path = self.get_intermediate_path(
                from_epoch,
                version=version,
                as_string=as_string,
            )
        else:
            path = self.results_load_dir / self.model_id
        if as_string:
            return str(path)
        return path

    def get_intermediate_path(self, epoch, version=None, as_string=True):
        """Get the path where intermediate weights for a specific epoch are saved

        Args:
            epoch (int): The epoch for which the weights are saved
            version (int, optional): An optional version number to provide. If None,
                the current version number for saving will be used. If version is provided and
                positive, that number will be used. If it is negative, the previous version number
                will be used (whatever the value provided). Defaults to None.
            as_string (bool, optional): Returns the result as a string instead of a pathlib.Path.
                Defaults to True.

        Returns:
            [type]: [description]
        """
        # * By default, use the save results dir (next version)
        res_dir = self.results_save_dir
        if version:
            if version > 0:
                # * A positive version number is provided, use this number
                res_dir = self.results_dir_root / str(version)
            else:
                # * The version number is negative, use previous version
                res_dir = self.results_dir_root / str(self.previous_version)
        path = res_dir / self.intermediate_save_dir / ("epoch_" + str(epoch))
        if as_string:
            return str(path)
        return path

    @property
    def model_id(self):
        if not self._model_id:
            self._model_id = self.name + self.resolve_id(self.id)
            # self.opts["model_id"] = self._model_id
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    def resolve_id(self, model_id):
        prefixes = self.id_prefixes
        to_replace = re.findall("\\{(.+?)\\}", model_id)
        res = {}
        for key in to_replace:
            mid = ""
            if prefixes:
                prefix = prefixes.get(key, prefixes.get("default", ""))
                mid += str(prefix)
            if self.DICT_PATH_SEPARATOR in key:
                value = common_utils.get_dict_path(
                    self.opts, key, key, sep=self.DICT_PATH_SEPARATOR
                )
            else:
                value = self.opts.get(key, key)
            mid += common_utils.list2str(value)
            res[key] = mid

        mid = model_id.format(**res)
        return mid

    # @property
    # def version(self):
    #     if self._version is None:
    #         v = self.opts.get("version", None)
    #         if not v:
    #             v = self.get_model_version(self.results_dir_root)
    #         self._version = v
    #     return self._version

    def get_last_version(self):
        version = 0
        path = self.results_dir_root
        if path.exists():
            for item in path.iterdir():
                if item.is_dir():
                    try:
                        # TODO: check if empty (or only databases)
                        if self.get("clean_empty_models", False):
                            is_empty = len(list(item.iterdir())) <= 1
                            if is_empty:
                                common_utils.print_info(
                                    "Found empty directory {} and clean_empty_models is True. Removing folder".format(
                                        item
                                    )
                                )
                                shutil.rmtree(item)
                                continue
                        res = int(item.name)
                        if res >= version:
                            version = res
                    except ValueError:
                        continue
        return version

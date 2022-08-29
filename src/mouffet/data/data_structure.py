from copy import deepcopy

from ..utils import common_utils


class DataStructure:
    """Inherit this class to define your data structure"""

    STRUCTURE = {
        "data": {"type": "data", "data_type": []},
        "tags": {"type": "tags", "data_type": []},
    }

    def __init__(self):
        self._structure = {}
        self.check_structure()

    @property
    def structure(self):
        return self._structure

    def check_structure(self):
        if isinstance(self.STRUCTURE, list):
            for element in self.STRUCTURE:
                self._structure[element] = {}
        elif isinstance(self.STRUCTURE, dict):
            self._structure = common_utils.deep_dict_update(
                self._structure, self.STRUCTURE, copy=True
            )

    def keys(self):
        return self.structure.keys()

    def get_structure_copy(self):
        return {
            key: deepcopy(self.structure[key].get("data_type", []))
            for key in self.structure.keys()
        }

    def get_extension(self, key):
        ext = self.structure[key].get("extension", "")
        if not ext:
            ext = "feather" if key.endswith("_df") else "pkl"
            self.structure[key]["extension"] = ext
        return ext

    def function_exists(self, func_name):
        return hasattr(self, func_name) and callable(getattr(self, func_name))

    def get_structure_function(self, key, name, *args, **kwargs):
        func_name = key + "_" + name
        if not self.function_exists(func_name):
            key_type = self.structure[key].get("type", "default")
            func_name = key_type + "_" + name
            if not self.function_exists(func_name):
                func_name = "default" + "_" + name
        func = getattr(self, func_name)
        return func(key, *args, **kwargs)

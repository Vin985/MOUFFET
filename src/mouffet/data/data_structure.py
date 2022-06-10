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

    def default_file_name(self, key, db_type, database):
        return db_type + "_" + key + "." + self.get_extension(key)

    def tags_file_name(self, key, db_type, database):
        return (
            db_type
            + "_"
            + key
            + "_"
            + database.class_type
            + "."
            + self.get_extension(key)
        )

    def get_file_name(self, key, db_type, database):
        func_name = key + "_file_name"
        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            func = getattr(self, func_name)
        else:
            key_type = self.structure[key].get("type", "default")
            func_name = key_type + "_file_name"
            if hasattr(self, func_name) and callable(getattr(self, func_name)):
                func = getattr(self, func_name)
            else:
                func = self.default_file_name
        return func(key, db_type, database)

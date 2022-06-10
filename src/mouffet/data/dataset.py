import pickle
from pathlib import Path

import feather
import pandas as pd

from ..utils import common_utils, file_utils
from .data_loader import DataLoader
from .data_structure import DataStructure


class Dataset(DataStructure):

    LOADERS = {"default": DataLoader}

    def __init__(self, db_type="test", database=None, paths=None, file_list=None):
        super().__init__()
        # self._paths = {}
        self.database = database
        self.db_type = db_type
        self.data = self.get_structure_copy()
        if database is not None:
            self.paths = self.update_paths(database.paths)
        self.file_list = file_list

    # @property
    # def paths(self):
    #     if self._paths:
    #         return self._paths
    #     else:
    #         self._paths = self.get_paths()

    def get_class_subfolder_path(self):
        """Function called if data is to be saved using a class subfolder.
        By default uses the value of the "class_type" option.

        Args:
            database (dict): The dictionary holding all option for the specific database.

        Returns:
            str: The class name
        """
        return self.database.class_type

    def get_subfolders(self):
        """Generate subfolders based on a list provided in the 'use_subfolders' option.
        For each item in the list, this function will try to call the
        get_itemname_folder_path(database) method from the DataHandler instance, where itemname is
        the name of the current item in the list. For example, if the item is "class", then the
        function will attempt to call the 'get_class_folder_path' method.
        If the method is not found, the option is skipped.
        Note that the called function should have the following signature:
        get_itemname_folder_path(database) -> str or pathlib.Path

        Args:
            database (dict): The dictionary holding all option for the specific database.

        Returns:
            pathlib.Path: a Path
        """
        res = Path("")
        subfolders = self.database.subfolders
        if subfolders:
            if isinstance(subfolders, str):
                subfolders = [{"type": subfolders}]
            for subfolder in subfolders:
                func_name = "_".join(["get", subfolder["type"], "subfolder_path"])
                if hasattr(self, func_name) and callable(getattr(self, func_name)):
                    res /= getattr(self, func_name)(subfolder)
                else:
                    print(
                        "Warning! No function found for getting the subfolder path for the '"
                        + subfolder
                        + "' option. Check if this is the correct value in the "
                        + "'use_subfolders' option or create a '"
                        + func_name
                        + "' function in your DataHandler instance."
                    )
        return res

    def get_subfolder_options(self, name):
        subfolder_opts = self.database.get("subfolders", [])
        for subfolder in subfolder_opts:
            if subfolder.get("type", "") == name:
                return subfolder
        return {}

    def get_save_dest_paths(self, dest_dir):
        """Create

        Args:
            dest_dir ([type]): [description]
            db_type ([type]): [description]
            subfolders ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = {}

        for key in self.structure.keys():
            res[key] = (
                dest_dir
                / self.get_subfolders()
                / self.get_file_name(key, db_type=self.db_type, database=self.database)
            )
        return res

    def update_paths(self, paths):
        if paths:
            paths["save_dests"][self.db_type] = self.get_save_dest_paths(
                paths["dest"][self.db_type]
            )
        return paths

    # def get_paths(self):
    #     paths = {}
    #     root_dir = self.database.root_dir

    #     paths["root"] = root_dir
    #     paths["data"] = {"default": self.database.data_dir}
    #     paths["tags"] = {"default": self.database.tags_dir}
    #     paths["dest"] = {"default": self.database.dest_dir / self.database["name"]}
    #     paths["file_list"] = {}
    #     paths["save_dests"] = {}

    #     by_type = self.database.get("data_by_type", False)
    #     if by_type:
    #         db_type_dir = file_utils.get_full_path(
    #             self.database[self.db_type + "_dir"], root_dir
    #         )
    #     else:
    #         db_type_dir = root_dir
    #     paths[self.db_type + "_dir"] = db_type_dir
    #     paths["data"][self.db_type] = file_utils.get_full_path(
    #         paths["data"]["default"], db_type_dir
    #     )
    #     paths["tags"][self.db_type] = file_utils.get_full_path(
    #         paths["tags"]["default"], db_type_dir
    #     )
    #     dest_dir = file_utils.get_full_path(paths["dest"]["default"], db_type_dir)
    #     paths["dest"][self.db_type] = dest_dir
    #     paths["file_list"][self.db_type] = self.database.get(
    #         self.db_type + "_file_list_path",
    #         dest_dir / (self.db_type + "_file_list.csv"),
    #     )
    #     paths["save_dests"][self.db_type] = self.get_save_dest_paths(dest_dir)
    #     return paths

    def generate(self, file_list, overwrite):
        loader_cls = self.LOADERS[self.database.get("loader", "default")]
        loader = loader_cls(self.get_structure_copy())
        loader.generate_dataset(
            self.database, self.paths, file_list, self.db_type, overwrite
        )
        self.save(loader.data)

    def save(self, data):
        if data:
            for key, value in data.items():
                path = self.paths["save_dests"][self.db_type][key]
                if path.suffix == ".pkl":
                    with open(
                        file_utils.ensure_path_exists(path, is_file=True), "wb"
                    ) as f:
                        pickle.dump(value, f, -1)
                        print("Saved file: ", path)
                elif path.suffix == ".feather":
                    if isinstance(value, pd.DataFrame):
                        value = value.reset_index(drop=True)
                        feather.write_dataframe(value, path)
                    else:
                        raise (
                            ValueError(
                                "Trying to write feather data from a source that is not a dataframe for key {}".format(
                                    key
                                )
                            )
                        )

    def get_loader(self):
        loader_cls = self.LOADERS[self.database.get("loader", "default")]
        loader = loader_cls(self.structure)
        return loader

    def load(self, load_opts=None):
        loader = self.get_loader()
        loader.load_dataset(self.paths, self.db_type, load_opts)
        self.data = loader.data

    def exists(self):
        for key in self.structure.keys():
            if not self.paths["save_dests"][self.db_type][key].exists():
                return False
        return True

    def summarize(self):
        return {}

    def get_raw_data(self):
        pass

    def get_ground_truth(self):
        pass

    def __getitem__(self, key):
        return self.data[key]

    def from_file_list(self, file_list, opts):
        pass

    def from_folder(self, folder_path, opts):
        pass

    def __iter__(self):
        loader = self.get_loader()
        if self.file_list is not None:
            for file_path in self.file_list:
                yield

    def copy(self):
        return self.__class__(self.db_type)

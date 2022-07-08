import pickle
from pathlib import Path

import feather
import pandas as pd

from ..utils import file_utils
from .data_loader import DataLoader
from .data_structure import DataStructure


class Dataset(DataStructure):

    LOADERS = {"default": DataLoader}

    def __init__(self, db_type="test", database=None, file_list=None):
        super().__init__()
        # self._paths = {}
        self.database = database
        self.db_type = db_type
        self.data = self.get_structure_copy()
        if database is not None:
            self.paths = self.update_paths(database.paths)
        self.file_list = file_list

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

    def generate(self, file_list, overwrite):
        loader_cls = self.LOADERS[self.database.get("loader", "default")]
        loader = loader_cls(self.get_structure_copy())
        loader.generate_dataset(
            self.database, self.paths, file_list, self.db_type, overwrite
        )
        self.save(loader.data)

    def save(self, data):
        """_summary_

        Args:
            data (_type_): _description_
        """
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
                                (
                                    "Trying to write feather data from a source that"
                                    + " is not a dataframe for key {}"
                                ).format(key)
                            )
                        )

    def get_loader(self):
        loader_cls = self.LOADERS[self.database.get("loader", "default")]
        loader = loader_cls(self.structure)
        return loader

    def load(self, load_opts=None):
        """_summary_

        Args:
            load_opts (_type_, optional): _description_. Defaults to None.
        """
        loader = self.get_loader()
        loader.load_dataset(self.paths, self.db_type, load_opts)
        self.data = loader.data

    def exists(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        for key in self.structure.keys():
            if not self.paths["save_dests"][self.db_type][key].exists():
                return False
        return True

    def summarize(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {}

    def get_raw_data(self):
        """_summary_"""
        pass

    def get_ground_truth(self):
        """_summary_"""
        pass

    def __getitem__(self, key):
        return self.data[key]

    # def from_file_list(self, file_list, opts):
    #     pass

    # def from_folder(self, folder_path, opts):
    #     pass

    # def __iter__(self):
    #     loader = self.get_loader()
    #     if self.file_list is not None:
    #         for file_path in self.file_list:
    #             yield

    def copy(self):
        return self.__class__(self.db_type)

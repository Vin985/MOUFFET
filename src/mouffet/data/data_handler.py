import csv
import pickle
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import feather
import pandas as pd

from ..options.database_options import DatabaseOptions
from ..utils import common as common_utils
from ..utils.file import ensure_path_exists, get_full_path, list_files
from .split import random_split
from .data_structure import DataStructure


class DataHandler(ABC):
    """
    A class that handles all data related business. While this class provides convenience functions,
    this should be subclassed
    """

    OPTIONS_CLASS = DatabaseOptions

    DATA_STRUCTURE = DataStructure()

    DB_TYPE_TRAINING = "training"
    DB_TYPE_VALIDATION = "validation"
    DB_TYPE_TEST = "test"

    SPLIT_FUNCS = {}

    def __init__(self, opts):
        self.opts = opts
        self.tmp_db_data = None
        self.databases = self.load_databases()

    def load_databases(self):
        """Loads all databases defined in the 'databases' option of the configuration file.

        Returns:
            dict: A dict where keys are the names of the databases and values are instances
            of the DataHandler.OPTIONS_CLASS that must be a subclass of
            mouffet.options.DatabaseOptions
        """
        global_opts = dict(self.opts)
        databases = global_opts.pop("databases")
        databases = {
            database["name"]: self.OPTIONS_CLASS(
                common_utils.deep_dict_update(dict(global_opts), database, copy=True)
            )
            for database in databases
        }
        return databases

    def duplicate_database(self, database):
        """Duplicates the provided database

        Args:
            database (instance of DataHandler.OPTIONS_CLASS): The database to duplicate

        Returns:
            mouffet.options.DatabaseOptions: The duplicated database
        """
        return self.OPTIONS_CLASS(
            common_utils.deep_dict_update(
                self.databases[database["name"]].opts, database, copy=True
            ),
            database,
        )

    def update_database(self, new_opts=None, name="", copy=True):
        """Updates a database with the options contained in new_opts.
        If 'name' is not provided, this function tries to get the name of the database to update
        from the 'name' key in new_opts.

        Args:
            new_opts (dict, optional): A dictionary containing the new value to update.
            Defaults to None.
            name (str, optional): The name of the database to update. Defaults to "".
            copy (bool, optional): If True, returns a copy of the original database.
            Defaults to True.

        Raises:
            AttributeError: Thrown when no database 'name' has been found.

        Returns:
            DataHandler.OPTIONS_CLASS: An options object with the values of the original database
            with updated values. Returns None if the database name was not found.
        """
        new_opts = new_opts or {}
        name = name or new_opts.get("name", "")
        if not name:
            raise AttributeError(
                "A valid database name should be provided, either with the name"
                + "or as a key in the new_opts dict"
            )
        if name in self.databases:
            return self.OPTIONS_CLASS(
                common_utils.deep_dict_update(
                    self.databases[name].opts, new_opts, copy=copy
                ),
                new_opts,
            )
        return None

    def get_class_subfolder_path(self, database):
        """Function called if data is to be saved using a class subfolder.
        By default uses the value of the "class_type" option.

        Args:
            database (dict): The dictionary holding all option for the specific database.

        Returns:
            str: The class name
        """
        return database.class_type

    def get_subfolders(self, database):
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
        subfolders = database.use_subfolders
        if subfolders:
            if isinstance(subfolders, str):
                subfolders = [subfolders]
            for subfolder_type in subfolders:
                func_name = "_".join(["get", subfolder_type, "subfolder_path"])
                if hasattr(self, func_name) and callable(getattr(self, func_name)):
                    res /= getattr(self, func_name)(database)
                else:
                    print(
                        "Warning! No function found for getting the subfolder path for the '"
                        + subfolder_type
                        + "' option. Check if this is the correct value in the "
                        + "'use_subfolders' option or create a '"
                        + func_name
                        + "' function in your DataHandler instance."
                    )
        return res

    def get_save_dest_paths(self, dest_dir, db_type, database, subfolders):
        """Create

        Args:
            dest_dir ([type]): [description]
            db_type ([type]): [description]
            subfolders ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = {}
        for key in self.DATA_STRUCTURE.keys():
            res[key] = (
                dest_dir
                / subfolders
                / self.DATA_STRUCTURE.get_file_name(
                    key, db_type=db_type, database=database
                )
            )
        return res

    def get_database_paths(self, database):
        paths = {}
        root_dir = database.root_dir

        subfolders = self.get_subfolders(database)

        paths["root"] = root_dir
        paths["data"] = {"default": database.data_dir}
        paths["tags"] = {"default": database.tags_dir}
        paths["dest"] = {"default": database.dest_dir / database["name"]}
        paths["file_list"] = {}
        paths["save_dests"] = {}

        for db_type in database.db_types:

            db_type_dir = get_full_path(database[db_type + "_dir"], root_dir)
            paths[db_type + "_dir"] = db_type_dir
            paths["data"][db_type] = get_full_path(
                paths["data"]["default"], db_type_dir
            )
            paths["tags"][db_type] = get_full_path(
                paths["tags"]["default"], db_type_dir
            )
            dest_dir = get_full_path(paths["dest"]["default"], db_type_dir)
            paths["dest"][db_type] = dest_dir
            paths["file_list"][db_type] = database.get(
                db_type + "_file_list_path", dest_dir / (db_type + "_file_list.csv")
            )
            paths["save_dests"][db_type] = self.get_save_dest_paths(
                dest_dir, db_type, database, subfolders
            )
        return paths

    @staticmethod
    def load_file_lists(paths, db_types=None):
        res = {}
        for db_type, path in paths["file_list"].items():
            if db_types and db_type in db_types:
                file_list = []
                with open(path, mode="r") as f:
                    reader = csv.reader(f)
                    for name in reader:
                        file_list.append(Path(name[0]))
                res[db_type] = file_list
                print("Loaded file: " + str(path))
        return res

    @staticmethod
    def save_file_list(db_type, file_list, paths):
        file_list_path = paths["dest"][db_type] / (db_type + "_file_list.csv")
        with open(ensure_path_exists(file_list_path, is_file=True), mode="w") as f:
            writer = csv.writer(f)
            for name in file_list:
                writer.writerow([name])
            print("Saved file list:", str(file_list_path))

    def get_data_file_lists(self, paths, database, db_types=None):
        res = {}
        db_types = db_types or database.db_types
        for db_type in db_types:
            res[db_type] = list_files(
                paths["data"][db_type], database.data_extensions, database.recursive
            )
        return res

    def split(self, paths, database):
        data_path = paths["data"]["training"]
        if not data_path.exists():
            raise ValueError(
                (
                    "Data path {} does not exist. Please provide a valid data folder to"
                    + "to split into test, training and"
                    + "validation subsets"
                ).format(data_path)
            )
        split_opts = database.get("split", None)
        if not split_opts:
            raise ValueError("Split option must be provided for splitting")
        split_func = self.SPLIT_FUNCS.get(database["name"], random_split)
        split_props = []
        # * Make test split optional
        test_split = split_opts.get("test", 0)
        if test_split:
            split_props.append(test_split)
        val_split = split_opts.get("validation", 0.2)
        split_props.append(val_split)
        splits = split_func(data_path, split_props, database.data_extensions)
        res = {}
        idx = 0
        if test_split:
            res["test"] = splits[idx]
            idx += 1
        elif "test" in database.db_types:
            res.update(self.get_data_file_lists(paths, database, db_types=["test"]))
        res["validation"] = splits[idx]
        res["training"] = splits[idx + 1]

        print([(k + " " + str(len(v))) for k, v in res.items()])
        return res

    def check_file_lists(self, database, paths, db_types=None):
        file_lists = {}
        msg = "Checking file lists for database {0}... ".format(database["name"])
        if db_types is None:
            file_list_paths = paths["file_list"].values()
        else:
            file_list_paths = [paths["file_list"][db_type] for db_type in db_types]
        file_lists_exist = all([path.exists() for path in file_list_paths])
        # * Check if file lists are missing or need to be regenerated
        if not file_lists_exist or database.generate_file_lists:
            print(msg + "Generating file lists...")
            file_lists = {}
            # * Check if we have a dedicated function to split the original data
            split_opts = database.get("split", None)
            if split_opts:
                file_lists = self.split(paths, database)
            else:
                file_lists = self.get_data_file_lists(paths, database)
            # * Save results
            for db_type, file_list in file_lists.items():
                self.save_file_list(db_type, file_list, paths)
        else:
            # * Load files
            print(msg + "Found all file lists. Now loading...")
            file_lists = self.load_file_lists(paths, db_types)
        return file_lists

    def load_classes(self, database):
        class_type = database.class_type
        classes_file = database.classes_file

        classes_df = pd.read_csv(classes_file, skip_blank_lines=True)
        classes = (
            classes_df.loc[classes_df["class_type"] == class_type]
            .tag.str.lower()
            .values
        )
        return classes

    @staticmethod
    @abstractmethod
    def load_raw_data(file_path, opts, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def load_tags(tags_dir, opts, *args, **kwargs):
        pass

    @abstractmethod
    def load_file_data(self, file_path, tags_dir, opts):
        """Load data for the file at file_path. This usually include loading the raw data
        and the tags associated with the file. This method usually include a call to
        load_raw_data() and load_tags(). This method should then fill the tmp_db_data attribute
        to save the intermediate results

        Args:
            file_path ([type]): [description]
            tags_dir ([type]): [description]
            opts ([type]): [description]
        """
        data, tags = [], []
        return data, tags

    def save_dataset(self, paths, db_type):
        if self.tmp_db_data:
            for key, value in self.tmp_db_data.items():
                path = paths["save_dests"][db_type][key]
                print(path)
                if path.suffix == ".pkl":
                    with open(ensure_path_exists(path, is_file=True), "wb") as f:
                        pickle.dump(value, f, -1)
                        print("Saved file: ", path)
                elif path.suffix == ".feather":
                    value = value.reset_index(drop=True)
                    feather.write_dataframe(value, path)

    def finalize_dataset(self):
        """Callback function called after data generation is finished but before it is saved
        in case some further action must be done after all files are loaded
        (e.g. dataframe concatenation)
        """
        pass

    @abstractmethod
    def load_data_options(self, database):
        return {}

    def generate_dataset(self, database, paths, file_list, db_type, overwrite):
        self.tmp_db_data = self.DATA_STRUCTURE.get_copy()
        print(self.tmp_db_data)
        print("Generating {} dataset for database {}".format(db_type, database["name"]))

        data_opts = self.load_data_options(database)

        for file_path in file_list:
            try:
                if not isinstance(file_path, Path):
                    file_path = Path(file_path)
                split = database.get("split", {})
                if split and db_type in split:
                    tags_dir = paths["tags"]["training"]
                else:
                    tags_dir = paths["tags"][db_type]
                intermediate = self.load_file_data(
                    file_path=file_path, tags_dir=tags_dir, opts=data_opts
                )

                if database.save_intermediates:
                    savename = (
                        paths["dest"][db_type] / "intermediate" / file_path.name
                    ).with_suffix(".pkl")
                    if not savename.exists() or overwrite:
                        with open(savename, "wb") as f:
                            pickle.dump(intermediate, f, -1)
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())
                self.tmp_db_data = None
        self.finalize_dataset()
        # Save all data
        print(
            len(self.tmp_db_data["infos"]),
            len(self.tmp_db_data["spectrograms"]),
            len(self.tmp_db_data["tags_linear_presence"]),
        )
        self.save_dataset(paths, db_type)
        self.tmp_db_data = None

    def check_dataset_exists(self, paths, db_type):
        for key in self.DATA_STRUCTURE.keys():
            if not paths["save_dests"][db_type][key].exists():
                return False
        return True

    def check_dataset(self, database, db_types=None):
        paths = self.get_database_paths(database)
        file_lists = self.check_file_lists(database, paths, db_types)
        db_types = db_types or database.db_types
        for db_type, file_list in file_lists.items():
            if db_types and db_type in db_types:
                print("Checking database:", database["name"], "with type", db_type)
                # * Overwrite if generate_file_lists is true as file lists will be recreated
                overwrite = database.overwrite or database.generate_file_lists
                if not self.check_dataset_exists(paths, db_type) or overwrite:
                    self.generate_dataset(
                        database, paths, file_list, db_type, overwrite
                    )

    def check_datasets(self, databases=None, db_types=None):
        databases = databases or self.databases.values()
        for database in databases:
            if isinstance(database, str):
                database = self.databases[database]
            self.check_dataset(database, db_types)

    def load_file(self, file_name):
        print("Loading file: ", file_name)
        if file_name.suffix == ".feather":
            return feather.read_dataframe(str(file_name))
        else:
            return pickle.load(open(file_name, "rb"))

    def merge_datasets(self, datasets):
        merged = self.DATA_STRUCTURE.get_copy()
        for dataset in datasets.values():
            for key in merged:
                if isinstance(dataset[key], list):
                    merged[key] += dataset[key]
                else:
                    merged[key].append(dataset[key])
        return merged

    def get_file_types(self, load_opts):
        file_types = load_opts.get("file_types", "all")
        if file_types == "all":
            file_types = self.DATA_STRUCTURE.keys()
        else:
            # * Make sure we only have valid keys
            file_types = [ft for ft in file_types if ft in self.DATA_STRUCTURE.keys()]
        return file_types

    def load_dataset(self, database, db_type, load_opts=None):
        load_opts = load_opts or {}
        file_types = self.get_file_types(load_opts)
        # * Get paths
        paths = self.get_database_paths(database)
        res = self.DATA_STRUCTURE.get_copy()

        for key in file_types:
            path = paths["save_dests"][db_type][key]
            if not path.exists():
                raise ValueError(
                    "Database file {} not found. Please run check_datasets() before".format(
                        str(path)
                    )
                )
            tmp = self.load_file(path)
            callback = load_opts.get("onload_callbacks", {}).get(key, None)
            if callback:
                tmp = callback(tmp)
            res[key] = tmp
        return res

    def get_database_options(self, name):
        return self.databases.get(name, None)

    def load_datasets(self, db_type, databases=None, by_dataset=False, load_opts=None):
        res = {}
        databases = databases or self.databases.values()
        # * Iterate over databases
        for database in databases:
            # * Only load data if the give db_type is in the database definition
            if db_type in database.db_types:
                print(
                    "Loading {0} data for database: {1}".format(
                        db_type, database["name"]
                    )
                )
                res[database["name"]] = self.load_dataset(database, db_type, load_opts)

        if not by_dataset:
            res = self.merge_datasets(res)
        return res

    @abstractmethod
    def get_summary(self, dataset):
        return {}

    def get_summaries(
        self, db_types=None, databases=None, detailed=True, load_opts=None
    ):
        res = {}
        databases = databases or self.databases.values()
        # * Iterate over databases
        for database in databases:
            db_types = db_types or database.db_types

            # * Only load data if the give db_type is in the database definition
            for db_type in db_types:
                if not db_type in database.db_types:
                    continue
                print(
                    "Generating summary for {0} data of database: {1}".format(
                        db_type, database["name"]
                    )
                )
                dataset = self.load_dataset(database, db_type, load_opts)

                summary = self.get_summary(dataset)

                if not database["name"] in res:
                    res[database["name"]] = {}

                res[database["name"]][db_type] = summary

        return res

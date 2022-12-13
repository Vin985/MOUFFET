import pickle
import traceback
from pathlib import Path
import feather

import pandas as pd

from ..utils import common_utils


class DataLoader:
    """Basic class for loading raw data into the dataset.
    By default, only the method :meth:`load_dataset` is called by the
    :class:`.data_handler.DataHandler` instance during the dataset generation call
    (by :meth:`.data_handler.DataHandler.generate_datasets`).
    A basic implementation of :meth:`load_dataset` is provided, however this method calls two
    other methods, :meth:`load_data_options` and :meth:`load_file_data`, that should be overriden
    since they do nothing by default.
    """

    CALLBACKS = {}

    def __init__(self, structure):
        self.data = structure

    def dataset_options(self, *args, **kwargs):
        common_utils.print_warning(
            (
                "WARNING! Calling load_data_options() method from the default DataLoader class which "
                + "does nothing. Please inherit this class and override this method for loading "
                + "any options relevant to the loading of the files of the dataset."
            )
        )
        return {}

    def load_file_data(self, *args, **kwargs):
        """Load data for the file at file_path. This usually include loading the raw data
        and the tags associated with the file. This method should then fill the tmp_db_data
        attribute to save the intermediate results

        Args:
            file_path ([type]): [description]
            tags_dir ([type]): [description]
            opts ([type]): [description]
        """
        common_utils.print_warning(
            (
                "Calling load_file_data() method from the default DataLoader class which "
                + "does nothing. Please inherit this class and override this method for loading "
                + "the files and tags of the dataset."
            )
        )
        data, tags = [], []
        return data, tags

    def finalize_dataset(self):
        """Callback function called after data generation is finished but before it is saved
        in case some further action must be done after all files are loaded
        (e.g. dataframe concatenation)
        """

    def generate_dataset(
        self, database, paths, file_list, db_type, missing=None, overwrite=False
    ):
        """[summary]

        Args:
            database ([type]): [description]
            paths ([type]): [description]
            file_list ([type]): [description]
            db_type ([type]): [description]
            overwrite ([type]): [description]
        """
        db_opts = self.dataset_options(database)
        split = database.get("split", {})
        if split and db_type in split:
            tags_dir = paths["tags"]["training"]
        else:
            tags_dir = paths["tags"][db_type]
        for file_path in file_list:
            try:
                if not isinstance(file_path, Path):
                    file_path = Path(file_path)
                intermediate = self.load_file_data(
                    file_path=file_path,
                    tags_dir=tags_dir,
                    opts=db_opts,
                    missing=missing,
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
                # self.data = None
        self.finalize_dataset(missing)

    def get_file_types(self, load_opts):
        file_types = load_opts.get("file_types", "all")
        if file_types == "all":
            file_types = self.data.keys()
        else:
            if isinstance(file_types, str):
                file_types = [file_types]
            # * Make sure we only have valid keys
            file_types = [ft for ft in file_types if ft in self.data.keys()]
        return file_types

    def load_dataset(self, paths, db_type, load_opts=None):
        # opts = common_utils.deepcopy(self.DEFAULT_LOADING_OPTIONS)
        # if load_opts is not None and isinstance(load_opts, dict):
        #     opts.update(load_opts)
        load_opts = load_opts or {}
        file_types = self.get_file_types(load_opts)
        # * Get paths

        for key in file_types:
            path = paths["save_dests"][db_type][key]
            if not path.exists():
                raise ValueError(
                    "Database file {} not found. Please run check_datasets() before".format(
                        str(path)
                    )
                )
            tmp = self.load_dataset_file(path)
            callback = self.CALLBACKS.get("onload", {}).get(key, None)
            if callback:
                tmp = callback(tmp)
            self.data[key] = tmp

    def load_dataset_file(self, file_name):
        print("Loading file: ", file_name)
        if file_name.suffix == ".feather":
            df = feather.read_dataframe(str(file_name))
            if df.empty:
                common_utils.print_warning(
                    "Warning, loaded dataset file {} is empty".format(file_name)
                )
            return df
        else:
            return pickle.load(open(file_name, "rb"))

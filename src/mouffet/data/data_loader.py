import pickle
import traceback
from pathlib import Path

import pandas as pd

from ..utils import common as common_utils


class DataLoader:
    """Basic class for loading raw data into the dataset.
    By default, only the method :meth:`load_dataset` is called by the
    :class:`.data_handler.DataHandler` instance during the dataset generation call
    (by :meth:`.data_handler.DataHandler.generate_datasets`).
    A basic implementation of :meth:`load_dataset` is provided, however this method calls two
    other methods, :meth:`load_data_options` and :meth:`load_file_data`, that should be overriden
    since they do nothing by default.
    """

    def __init__(self, structure):
        self.data = structure.get_copy()

    def load_data_options(self, *args, **kwargs):
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

    def load_dataset(self, database, paths, file_list, db_type, overwrite):
        """[summary]

        Args:
            database ([type]): [description]
            paths ([type]): [description]
            file_list ([type]): [description]
            db_type ([type]): [description]
            overwrite ([type]): [description]
        """
        db_opts = self.load_data_options(database)
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
                    file_path=file_path, tags_dir=tags_dir, opts=db_opts
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
                self.data = None
        self.finalize_dataset()

import pickle
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataLoader(ABC):
    def __init__(self, structure):
        self.data = structure.get_copy()

    @abstractmethod
    def load_data_options(self, database):
        return {}

    @abstractmethod
    def load_file_data(self, file_path, tags_dir, opts):
        """Load data for the file at file_path. This usually include loading the raw data
        and the tags associated with the file. This method should then fill the tmp_db_data
        attribute to save the intermediate results

        Args:
            file_path ([type]): [description]
            tags_dir ([type]): [description]
            opts ([type]): [description]
        """
        data, tags = [], []
        return data, tags

    def finalize_dataset(self):
        """Callback function called after data generation is finished but before it is saved
        in case some further action must be done after all files are loaded
        (e.g. dataframe concatenation)
        """
        pass

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

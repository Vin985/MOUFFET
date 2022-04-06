from mouffet.data import DataHandler
from mouffet.data.data_structure import DataStructure
from mouffet.evaluation import EvaluationHandler
import tensorflow_datasets as tfds
import tensorflow as tf


class TFDatasetStructure(DataStructure):
    STRUCTURE = {"data": {"type": "data"}, "metadata": {"type": "metadata"}}


class TFExampleDataHandler(DataHandler):
    DATA_STRUCTURE = TFDatasetStructure()

    def __init__(self, opts):
        super().__init__(opts)

    def check_datasets(self, databases=None, db_types=None):
        return True

    def get_split_strings(self, database):
        splits_strings = database.get("split_strings", {})
        if not splits_strings:
            splits = database.split

            start = 0
            end = 0
            for db_type in [
                self.DB_TYPE_TRAINING,
                self.DB_TYPE_VALIDATION,
                self.DB_TYPE_TEST,
            ]:
                split_str = ""
                if db_type in splits:
                    split_val = splits[db_type]
                    end += split_val
                    if start:
                        split_str += str(start) + "%"
                    split_str += ":"
                    if end < 100:
                        split_str += str(end) + "%"
                    start += split_val
                splits_strings[db_type] = split_str
            database.add_option("split_strings", splits_strings)
        return splits_strings

    def load_tf_flowers(self, database, db_type, load_opts):
        split_strings = self.get_split_strings(database)
        split_str = "train[" + split_strings[db_type] + "]"
        ds, metadata = tfds.load(
            "tf_flowers",
            split=[split_str],
            with_info=True,
            as_supervised=True,
        )
        struct = self.DATA_STRUCTURE.get_copy()
        struct["data"] = ds[0]
        struct["metadata"] = metadata
        return struct

    def load_dataset(self, database, db_type, load_opts=None):
        db_name = database.name
        dataset = getattr(self, "load_" + db_name)(database, db_type, load_opts)
        return dataset

    def merge_datasets(self, datasets):
        merged = self.DATA_STRUCTURE.get_copy()
        for dataset in datasets.values():
            for key in merged:
                if isinstance(dataset[key], list):
                    merged[key] += dataset[key]
                elif isinstance(dataset[key], tf.data.Dataset):
                    if merged[key]:
                        merged[key] = merged[key].concatenate(dataset[key])
                    else:
                        merged[key] = dataset[key]

                else:
                    merged[key].append(dataset[key])
        return merged

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

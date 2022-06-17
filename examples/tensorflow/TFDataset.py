from mouffet.data import Dataset, ALL_DB_TYPES
import tensorflow_datasets as tfds


class TFDataset(Dataset):
    STRUCTURE = {"data": {"type": "data"}, "metadata": {"type": "metadata"}}

    def get_split_strings(self):
        splits_strings = self.database.get("split_strings", {})
        if not splits_strings:
            splits = self.database.get("split", {})

            start = 0
            end = 0
            for db_type in ALL_DB_TYPES:
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
            self.database.add_option("split_strings", splits_strings)
        return splits_strings

    def load_tf_flowers(self, load_opts=None):
        split_strings = self.get_split_strings()
        split_str = "train[" + split_strings[self.db_type] + "]"
        ds, metadata = tfds.load(
            "tf_flowers",
            split=[split_str],
            with_info=True,
            as_supervised=True,
        )
        self.data = self.get_structure_copy()
        self.data["data"] = ds[0]
        self.data["metadata"] = metadata

    def load(self, load_opts=None):
        db_name = self.database.name
        dataset = getattr(self, "load_" + db_name)(load_opts)
        return dataset

    def get_ground_truth(self):
        return self.data["data"]["labels"]

    def get_raw_data(self):
        return self.data["data"]["images"]

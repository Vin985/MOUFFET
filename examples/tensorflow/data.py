import tensorflow_datasets as tfds
from mouffet.data import ALL_DB_TYPES, Database, DataHandler, Dataset

import tensorflow as tf


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

    def load(self, load_opts=None):
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
        return self.data


class TFDatabase(Database):
    DATASET = TFDataset

    def check_dataset(self, database, db_types=None):
        return True

    def get_paths(self):
        return {}


class TFExampleDataHandler(DataHandler):
    DATABASE_CLASS = TFDatabase

    def __init__(self, opts):
        super().__init__(opts)

    def merge_datasets(self, datasets):
        merged = None
        for dataset in datasets.values():
            if not merged:
                merged = dataset.get_structure_copy()
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

    def prepare_dataset(self, dataset, opts):
        """Prepare data before training the model. This function is automatically called
        after loading the datasets

        Args:
            data (_type_): The data to prepare. Here it is a Tensorflow dataset

        Returns:
            the prepared data
        """

        # * Resize and rescale all datasets.
        ds = dataset.data["data"]

        img_size = opts.get("img_size", 128)
        resize_and_rescale_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Resizing(img_size, img_size),
                tf.keras.layers.Rescaling(1.0 / 255),
            ]
        )

        data_augmentation_layers = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(opts.get("flip", "horizontal_and_vertical")),
                tf.keras.layers.RandomRotation(opts.get("rotation", 0.2)),
            ]
        )

        ds = ds.map(
            lambda x, y: (resize_and_rescale_layers(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if opts.get("shuffle_data", True):
            ds = ds.shuffle(1000)

        # * Batch all datasets.
        ds = ds.batch(opts.get("batch_size", 32))

        # * Use data augmentation.
        if opts.get("augment_data", True):
            ds = ds.map(
                lambda x, y: (data_augmentation_layers(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # * Use buffered prefetching on all datasets.
        dataset.data["data"] = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

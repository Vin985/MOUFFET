from mouffet.models import DLModel

import tensorflow as tf
from tensorflow.keras import layers


class TFSequentialModel(DLModel):
    def get_ground_truth(self, data):
        return data["labels"]

    def get_raw_data(self, data):
        return data["images"]

    def classify(self, data, sampler=None):
        return self.model.predict(data)


class SimpleModel(TFSequentialModel):
    def __init__(self, opts=None):
        super().__init__(opts)

        img_size = self.opts.get("img_size", 128)
        self.resize_and_rescale_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Resizing(img_size, img_size),
                tf.keras.layers.Rescaling(1.0 / 255),
            ]
        )

        self.data_augmentation_layers = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
            ]
        )

    def create_model(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(5),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self, training_data, validation_data):
        self.create_model()
        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.opts.get("n_epochs", 3),
        )
        # Return information saved in callbacks
        res = history.history
        res.update(history.params)
        return res

    def prepare_data(self, data):
        """Prepare data before training the model. This function is automatically called
        after loading the datasets

        Args:
            data (_type_): The data to prepare. Here it is a Tensorflow dataset

        Returns:
            the prepared data
        """

        # Resize and rescale all datasets.
        ds = data["data"]
        ds = ds.map(
            lambda x, y: (self.resize_and_rescale_layers(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.opts.get("shuffle_data", True):
            ds = ds.shuffle(1000)

        # Batch all datasets.
        ds = ds.batch(self.opts.get("batch_size", 32))

        # Use data augmentation only on the training set.
        if self.opts.get("augment_data", True):
            ds = ds.map(
                lambda x, y: (self.data_augmentation_layers(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def save_weights(self, path=None):
        if not path:
            path = str(self.opts.results_save_dir / self.opts.model_id)
        self.model.save_weights(path)  # pylint: disable=no-member

    def load_weights(self):
        print("Loading pre-trained weights")
        self.model.load_weights(  # pylint: disable=no-member
            self.opts.get_weights_path()
        )

    def predict(self, x):
        return tf.nn.softmax(self.model(x, training=False)).numpy()

from mouffet.models import DLModel

import tensorflow as tf


class SimpleTFModel(DLModel):
    def __init__(self, opts=None):
        super().__init__(opts)

    def create_model(self):
        model = tf.keras.Sequential(
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
        model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def train(self, training_data, validation_data):
        early_stopping = self.opts.get("early_stopping", {})
        callbacks = []
        if early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    # * Stop training when `val_loss` is no longer improving
                    monitor=early_stopping.get("monitor", "val_loss"),
                    # * "no longer improving" being defined as "no better than 1e-2 less"
                    min_delta=early_stopping.get("min_delta", 1e-2),
                    # * "no longer improving" being further defined as "for at least 2 epochs"
                    patience=early_stopping.get("patience", 2),
                    verbose=early_stopping.get("verbose", 1),
                    restore_best_weights=early_stopping.get(
                        "restore_best_weights", True
                    ),
                )
            )
        self.model = self.create_model()
        history = self.model.fit(
            training_data["data"],
            validation_data=validation_data["data"],
            epochs=self.opts.get("n_epochs", 3),
            callbacks=callbacks,
        )
        # * Return information saved in callbacks
        res = history.history
        res.update(history.params)
        return res

    def save_weights(self, path=None):
        if not path:
            path = str(self.opts.results_save_dir / self.opts.model_id)
        self.model.save_weights(path)  # pylint: disable=no-member

    def load_weights(self):
        print("Loading pre-trained weights")
        self.model.load_weights(  # pylint: disable=no-member
            self.opts.get_weights_path()
        ).expect_partial()

    def predict(self, x):
        return tf.nn.softmax(self.model.predict(x)).numpy()

    def classify(self, data, sampler=None):
        return self.model.predict(data)

#%%
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import numpy as np

batch_size = 32
IMG_SIZE = 180
AUTOTUNE = tf.data.AUTOTUNE


(train_ds, val_ds, test_ds), metadata = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True,
)
num_classes = metadata.features["label"].num_classes
class_names = metadata.features["label"].names

# get_label_name = metadata.features["label"].int2str

#%%

# image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# _ = plt.title(get_label_name(label))

resize_and_rescale = tf.keras.Sequential(
    [layers.Resizing(IMG_SIZE, IMG_SIZE), layers.Rescaling(1.0 / 255)]
)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)


def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)


model = tf.keras.Sequential(
    [
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(train_ds, validation_data=val_ds, epochs=3)

#%%

model.evaluate(test_ds)
preds = model.predict(test_ds)

scores = tf.nn.softmax(preds)
score1 = scores[1]
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score1)], 100 * np.max(score1)
    )
)

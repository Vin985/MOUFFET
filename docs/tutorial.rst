Tutorial: Creating image classifiers with Tensorflow
####################################################


Implementing a full example
===========================

To show the capabilities of Mouffet, we present here how to easily and robustly create several image
classifiers using Mouffet and deep learning. To that end, we adapted the data augmentation tutorial
found on the Tensorflow website (https://www.tensorflow.org/tutorials/images/data_augmentation)
that aims to develop a simple deep learning model to classify images of flowers.
  
Please note that this example aims only to showcase Mouffet features and not to be a deep learning
tutorial.
  
As Mouffet is independent of the implementation of the model, no prior knowledge of working with
Tensorflow or even deep learning is required. Instead we will show which Python classes are
required and how to use the configuration files to train and evaluate several models.
We will show how by using Mouffet, we can easily investigate the effect of multiple values of 
data augmentation. All steps describing how to run the example can be found at
https://github.com/Vin985/mouffet/blob/main/EXAMPLE.md 

Goal
====

Tensorflow's tutorial aims to develop a simple deep learning model to classify images of five types of flowers
using data augmentation techniques such as flipping and rotating the images. Data augmentation
is a common practice in deep learning to increase the number of available examples, reduce
overfitting, and increase training accuracy. After training, for each class of flower, 
the model gives a probability that an image belongs to this class. Model performance is then 
assessed by selecting the class with the highest probability and comparing it with the ground truth.

In this example, we will adapt that model using Mouffet’s workflow to train five different models 
using each different parameters of data augmentation. We will define a custom evaluator that allows the 
user to choose a probability threshold for an image to be deemed well classified. This means that, 
if the classification probability of the best class is under the selected threshold, the image will 
be assigned the label “Unsure”. Mouffet’s plotting capabilities will be used to compare the performance 
of each model with different tolerance thresholds and to generate confusion matrices for each 
evaluation scenario.


Installation
============

Dependencies
------------

For the following example, the following additional libraries will be required: 
“tensorflow”, “tensorflow-datasets” , “scikit-learn” (to easily calculate evaluation metrics and plot
confusion matrices), and “plotnine" (a plotting library implementing the grammar of graphics used
in the ggplot2 R package).

To install them please enter the following line in a command line or terminal:

.. code-block:: bash

    > pip install tensorflow tensorflow-datasets plotnine sklearn

Download the exemple
--------------------

The full example files can be downloaded at this address: 

    https://github.com/Vin985/mouffet/blob/main/examples/flowers_example.zip

You just need to extract the archive where you desire.
Then open a terminal and go to the 'flowers' folder of the example.



Data management
===============

Code for all classes about data management can be found in the data.py file of the example.

Data source
-----------

The data used in this example is taken from the "tf_flowers" database that can be downloaded and
accessed via the tensorflow-datasets package. This database contains 3670 images of flowers
separated among 5 classes (dandelion, daisy, tulips, sunflowers and roses).

Data configuration
------------------

The data configuration file for this example is really simple. We define a database named 
"tf _flowers" in the "databases" options. Since the data will be downloaded from internet, 
there is no need to define paths here. We only define how the datasets will be split. 
Here we decide to put aside 80% of the dataset for training, 10% for validation and 10% for testing.

.. code-block:: yaml

    # List all databases
    databases:
    - name: 'tf_flowers' # Provide the name of the database
      split: # Define how to split the data for training, validation and test purposes
        training: 80
        validation: 10
        test: 10


Data representation
-------------------

In Python, data is represented by two classes: :doc:`mouffet.data.Database<api/data/database>` and 
:doc:`mouffet.data.Dataset<api/data/dataset>`.



Dataset
~~~~~~~

The :doc:`mouffet.data.Dataset<api/data/dataset>` class describes the structure of the data and
provides convenience functions for loading and saving datasets.
Here we inherit it to create the **FlowersDataset** class.

.. code-block:: python


    class FlowersDataset(Dataset):
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


The most important things to define are:

 - **STRUCTURE** (inherited from :doc:`mouffet.data.DataStructure<api/data/data_structure>`): A property
   that defines how the data is organized. When a dataset is loaded, it will create an object with 
   this structure and fill it with the appropriate data. It is usually a simple python dict and its 
   content are defined by the user. Here we create two entries "data" and "metadata" as this is how 
   the data is provided when loading the "tf_flowers" dataset.
 - :meth:`load()<mouffet.data.dataset.Dataset.load>`: The method that actually loads the dataset. 
   The main content of this function actually just calls the tensorflow_datasets.load() function 
   with the database name and then fills a copy of the STRUCTURE object with the data loaded.

Database
~~~~~~~~

The :doc:`mouffet.data.Database<api/data/database>` class stores information about a database, 
such as the paths where the data is located and provide functions to manage datasets such as
checking if a dataset exists or splitting the data into the appropriate datasets. 
Here we create the **FlowersDatabase**.

.. code-block:: python

    class FlowersDatabase(Database):
        DATASET = FlowersDataset

        def check_dataset(self, database, db_types=None):
            return True

        def get_paths(self):
            return {}


One of the most important things to change here is the **DATASET** property that defines the class to
use to create datasets. Here it is the FlowersDataset class we previously defined. In this example, 
we bypass the :meth:`check_dataset()<mouffet.data.database.Database.check_dataset>` and
:meth:`get_paths()<mouffet.data.database.Database.get_paths>` methods as they are irrelevant since our 
dataset is dowladed from the internet and not present on our hard drive.


.. _data_handler:

Data manipulation
-----------------

In Mouffet, data is handled with an implementation of the
:doc:`mouffet.data.DataHandler<api/data/data_handler>` class, called **FlowersDatahandler** here. 
This class mainly helps managing databases and datasets. 
One of its most important task is to prepare datasets training or evaluation, i.e. modify the raw
data to fit the model.

.. code-block:: python

    class FlowersDataHandler(DataHandler):
        DATABASE_CLASS = FlowersDatabase

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
            seed = opts.get("seed", None)

            img_size = opts.get("img_size", 128)
            resize_and_rescale_layers = tf.keras.Sequential(
                [
                    tf.keras.layers.Resizing(img_size, img_size),
                    tf.keras.layers.Rescaling(1.0 / 255),
                ]
            )

            data_augmentation_layers = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip(
                        opts.get("flip", "horizontal_and_vertical"), seed=seed
                    ),
                    tf.keras.layers.RandomRotation(opts.get("rotation", 0.2), seed=seed),
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


Any modification to a dataset should be done in the :meth:`prepare_dataset()<mouffet.data.data_handler.DataHandler.prepare_dataset>` method.
We add in this function all the data augmentation features,
as well as the shuffle and batch preparation logic for training and evaluation. Note that in this
method, values for variables of interest are taken from the "opts" object via options we defined
such as the angle of rotation via the **rotation** option or the way images should be flipped via
the **flip** option.

Model
=====


Models are at the core of the package as they are created during training and called during
evaluation. To define a model, we need to inherit the :doc:`mouffet.model.Model class<api/models/model>`.
Since the models we are training are deep learning models, we can actually inherit the
:doc:`mouffet.model.DLModel class<api/models/dlmodel>` class which provides additional convenience
functions. 

We create a FlowersClassifier class that essentially copies the tutorial
We implement the following methods:

 - **create_model()**: This is where we define how the model works. With deep learning, this is where we define the layers of the model
 - **train()**: The method called to train the model. Note that here we added logic for early
   stopping.
 - **save_weights()/load_weights()**: methods called to save/load the weights of the model, i.e. 
   the result of the training
 - **predict()**: The method called to actually use the model

.. code-block:: python

    class FlowersClassifier(DLModel):
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
                if not isinstance(early_stopping, dict):
                    early_stopping = {}
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        # * Stop training when `val_loss` is no longer improving
                        monitor=early_stopping.get("monitor", "val_loss"),
                        # * "no longer improving" being defined as "no better than 1e-2 less"
                        min_delta=early_stopping.get("min_delta", 1e-2),
                        # * "no longer improving" being further defined as "for at least 2 epochs"
                        patience=early_stopping.get("patience", 3),
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
                epochs=self.opts.get("n_epochs", 20),
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


Training
========

Training configuration
----------------------

.. code-block:: yaml

    ###########
    ## Paths ##
    ###########

    # Where to find the data configuration file
    data_config: "config/flowers/data_config.yaml"
    # Where to save the models
    model_dir: "results/models"

    ###############
    ## Databases ##
    ###############

    # List the databases to use. The training and validation datasets will be used.
    databases: ["tf_flowers"]

    ###########
    ## Model ##
    ###########

    # All options below will be applied to all models unless overwritten

    # Class of the model
    class: ".models.FlowersClassifier"
    # Maximum number of iterations for training
    n_epochs: 20
    # Shuffle the data for each epoch
    shuffle_data: True
    # Number of images passed to the model at the same time
    batch_size: 32
    # Augment the data
    augment_data: True

    # Stop training if no improvement is noticed
    early_stopping:
    patience: 3

    # Do not train models that already exist
    skip_trained: True

    ###############
    ## Scenarios ##
    ###############

    # Training scenarios 
    scenarios:
    # Do not augment the data
    - name: "no_augment"
        augment_data: False
    # Augment the data with different values
    - name: "augment"
        # Suffix to add to the name depending on the values of the variables
        suffix: "_rot-{rotation}_flip-{flip}"
        # Rotation values
        rotation: [0.2, 0.3]
        # Filp values
        flip: ["horizontal", "horizontal_and_vertical"]


Here, besides the paths to the data configuration file and where to save the models, we define 
global options for the models such as the number of training iterations. We specify that we want 
to shuffle and augment the data and that we want the training to stop as soon as there seem to be 
no more improvements (early stopping). 

The most important part is the "scenarios" options where we define all the scenarios we want to
perform. 

.. code-block:: yaml

    scenarios:
    # Do not augment the data
    - name: "no_augment"
        augment_data: False
    # Augment the data with different values
    - name: "augment"
        # Suffix to add to the name depending on the values of the variables
        suffix: "_rot-{rotation}_flip-{flip}"
        # Rotation values
        rotation: [0.2, 0.3]
        # Filp values
        flip: ["horizontal", "horizontal_and_vertical"]

First we define a scenario with no data augmentation where we override the "augment_data" option
specifically for this scenario. Then we define augmentation scenarios where we change the values 
for data augmentation. Note that for the **rotation** option, we define a list of values, as is the
case for the **flip** option. Mouffet will actually iterate over each of these lists and create
all the permutations possible for these values. So we will actually train 4 models with different
values for rotation and flip.

To keep track of these values, we add a suffix to our model name. Each variable name between brackets
will be replaced by the actual value of the variable during the training. This results in models
named, for example **augment_rot-0.2_flip-horizontal**

Training handler
----------------

Training in mouffet is handled by an instance of the class :doc:`mouffet.training.TrainingHandler<api/training/training_handler>`. 
This class handles logic related to the training options, creates training scenarios and for each 
of these scenarios, loads the right data, creates the appropriate model and then calls the 
:meth:`train()<mouffet.model.Model.train>` method of the model with the data. 
During training, only training and validation datasets will be used. Each model is then saved, as 
well as the options that led to its creation. This ensure full traceability and reproductiblity.


Since this example is quite simple, here we do not need to implement a new training handler.
We use the one provided by with Mouffet instead. Training occurs in the **training.py** file.

.. code-block:: python


    if __name__ == "__main__":
        trainer = TrainingHandler(
            opts_path="config/flowers/training_config.yaml",
            dh_class=FlowersDataHandler,
        )
        trainer.train()



We create an instance of :doc:`mouffet.training.TrainingHandler<api/training/training_handler>`
by providing the path to the training configuration file and the class of the 
:ref:`data handler<data_handler>` used to manage the data. Once the object is created, all 
that remains is to call the train() method that will launch all training scenarios.
By default, Mouffet keeps a copy of each version of a model trained and statistics for this model
if training was successful.

To launch training, in a command line we type:

.. code-block:: bash
    
    > python training.py

After training, we should get 5 models that can be found in the folder specified in the **model_dir**
option of the configuration file.

Evaluation
==========

Evaluation configuration
------------------------

Once more, we define paths where to find models, and databases and where to save the results.

.. code-block:: yaml

    ###########
    ## Paths ##
    ###########

    # Where to find the data configuration file
    data_config: "config/flowers/data_config.yaml"
    # Where to save predictions
    predictions_dir: "results/predictions"
    # Where to save evaluation results
    evaluation_dir: "results/evaluation"

    ###############
    ## Databases ##
    ###############

    # List databases to use during the evaluation. Only the test datasets will be used. databases
    # without test datasets will be skipped
    databases:
    - name: tf_flowers

    ###########
    ## Plots ##
    ###########

    # Draw plots with results from all evaluation
    draw_global_plots: True
    # List of plots to draw
    global_plots: ["accuracy_f1"]

    ## Global options for all evaluators
    evaluators_options:
    # Draw plots with results from each evaluator
    draw_plots: True
    # Which plots to draw
    plots: ["confusion_matrix"]

    ## Options for specific plots
    plot_options:
    confusion_matrix:
        # Which package to use to draw the plots
        package: mouffet.plotting.sklearn

    ################
    ## Evaluators ##
    ################

    # List evaluators to use
    evaluators: 
    - type: "custom"
        # Change thresholds values
        scenarios:
        threshold: [-1, 0.3, 0.5, 0.75]

    ############
    ## Models ##
    ############

    ## Global options for all models
    models_options:
    # where to find the models
    model_dir: "results/models"
    # Do not shuffle data for evaluation
    shuffle_data: False
    # Do not augment data for evaluation
    augment_data: False

    # List all models used for evaluation
    models:
    # The entries below are copied from the training_config.yaml file. This is not used during the runs
    - name: "no_augment"
        class: ".models.FlowersClassifier"
    - name: "augment"
        class: ".models.FlowersClassifier"
        suffix: "_rot-{rotation}_flip-{flip}"
        scenarios: 
        rotation: [0.2, 0.3]
        flip: ["horizontal", "horizontal_and_vertical"]


Three options are essential here:

 - **databases**: defines which databases will be used.
 - **models**: defines which models will be evaluated. Here we simply copy the training
   configuration file.
 - **evaluators**: defines the evaluators to be used. Each evaluator needs a unique name that will
   be defined during registration (See below for more information). Here we set 4 different values
   for the **threshold** option inside a **scenarios** keyword.

With **models_options** and **evaluators_options**, we define options that will be applied to all
models and evaluators respectively. Here we do not want to augment the data and tell the evaluators
to plot confusion matrixes. We also define global plots that will be generated from all the results.

Evaluator
---------

We create a subclass of :doc:`mouffet.evaluator.Evaluator<api/evaluation/evaluator>` called 
CustomEvaluator in the file **evaluators.py**. In its **evaluate()** method we added a **threshold**
option taken from the configuration file that behaves like this:

 - If **threshold** == -1, the evaluator behaves like the original tutorial, i.e., 
   the selected class is the one with the highest given probability.
 - If 0 <= **threshold** <= 1, and the highest probability is above the threshold, its associated 
   class is returned. Otherwise, the image is attributed to the class *"Unsure"*.
 - Otherwise, an error is raised.

.. code-block:: python

    class CustomEvaluator(Evaluator):
        def get_label_names(self, columns, metadata):
            res = []
            for x in columns:
                if x == -1:
                    res.append("Unsure")
                else:
                    res.append(metadata.features["label"].int2str(int(x)))
            return res

        def plot_confusion_matrix(self, data, options, infos):
            cm = confusion_matrix(data["labels"], data["predictions"])
            cm_plot = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=data["label_names"]
            )
            cm_plot.plot()
            plt.title(
                "Confusion matrix for model '{}'\nwith threshold: {}".format(
                    infos["model"], options["threshold"]
                ),
                fontweight="bold",
                fontsize=12,
            )
            plt.xlabel("Predicted class", fontweight="bold")
            plt.ylabel("True class", fontweight="bold")
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            return cm_plot

        def evaluate(self, data, options, infos):
            res = {}
            preds, labels, meta = data
            # * Get class with the best prediction score
            thresh = options.get("threshold", -1)
            if thresh > 1:
                raise ValueError(
                    "The option 'threshold' should only take values between 0 and 1 or -1"
                )
            npreds = preds.to_numpy()
            top_class = npreds.argmax(axis=1)
            if thresh != -1:
                unsolved = npreds.max(axis=1) <= thresh
                top_class[unsolved] = -1

            # * Get label names
            label_names = self.get_label_names(np.unique(top_class), meta)
            # * get classification report from sklearn
            cr = classification_report(
                labels,
                top_class,
                output_dict=True,
                target_names=label_names,
            )
            equals = (labels == top_class).astype(int)
            # * Print report
            common_utils.print_warning(
                classification_report(labels, top_class, target_names=label_names)
            )

            res["stats"] = pd.DataFrame([cr])
            res["matches"] = pd.DataFrame(equals)

            if options.get("draw_plots", False):
                res["plots"] = self.draw_plots(
                    data={
                        "labels": labels,
                        "predictions": top_class,
                        "label_names": label_names,
                    },
                    options=options,
                    infos=infos,
                )

            return res

We used the method `classification_report() <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>`_ 
from the package scikit-learn to calculate metrics such as precision, recall and f1-score for each
class and globally. We also added the **plot_confusion_matrix()** method, which plots confusion
matrices using the `scikit-learn package <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay>`_.

Performing the evaluation
-------------------------

Evaluation is done in the **evaluation.py** file. 

.. code-block:: python

    EVALUATORS.register_evaluator("custom", CustomEvaluator)

    if __name__ == "__main__":
        evaluator = FlowersEvaluationHandler(
            opts_path="config/flowers/evaluation_config.yaml",
            dh_class=FlowersDataHandler,
        )

        res = evaluator.evaluate()


The concept is the same as for training, we create the FlowersEvaluationHandler and just call the
**evaluate()** function. The main difference is that we need to register our evaluator by giving it
a unique name to be able to use it. That name is the one put in the configuration file in the 
**evaluators** option.

To launch the evaluation, we type in the command line

.. code-block:: bash

    > python evaluation.py

Once the evaluation was done, we get the following results:
 - In the *results/predictions* folder, the predictions for each model on each database, saved in
   the feather format. A **predictions_stats.csv** file that compiles information about how the
   predictions were generated (time spent, options used, etc.) can also be found.
 - In the *results/evaluation* folder, we find the evaluation results. Here this consists in: 
   a csv file that contains all results - including the options used to generate them - in a table
   format; a pdf with all confusion matrixes plots and a pdf with the accuracy/f1 score plots
   generated from all results. Note that all files are time-stamped and sorted by date to keep
   track of when the evaluation was performed.

Launching a run
===============

To avoid having to launch separately the training and evaluation, we can define a run that will do
both for us. Runs are named, and all the configuration files should be put inside a folder with the
run name. Here we put them in the **flowers** folder. Then inside the file **run.py**, we create a 
RunHandler that takes all handlers as arguments.

.. code-block:: python

    EVALUATORS.register_evaluator("custom", CustomEvaluator)

    run_handler = RunHandler(
        handler_classes={
            "training": TrainingHandler,
            "data": FlowersDataHandler,
            "evaluation": FlowersEvaluationHandler,
        },
        default_args={
            "run_dir": "config",
        },
    )

    run_handler.launch_runs()

We then called the **run.py** file by passing the name of our folder as a command line argument. 

.. code-block:: bash

    > python run.py flowers

We get the same results as if we launched the processes manually, except that results are now saved
in the **results/runs/flowers** folder.

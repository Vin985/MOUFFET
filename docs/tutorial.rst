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

Models are at the core of the package as they are created during training and called during evaluation. To define a model, we need to inherit the mouffet.model.Model class. Since the models we are training are deep learning models, we can actually inherit the provided mouffet.model.DLModel class which provides additional convenience functions for deep learning models. Thus with create the class TFModel simple model with the following methods:
create_model(): This is where we define how the model works. With deep learning, this is where we define the layers of the model
train(): The method called to train the model.
save_weights()/load_weights(): methods called to save/load the weights of the model, i.e. the result of the training
predict(): The method called to actually use the model

Training
========

Training configuration
----------------------

Here, besides the paths to the data configuration file and where to save the models, we define global options for the models such as the number of training iterations. We specify that we want to shuffle and augment the data and that we want the training to stop as soon as there seem to be no more improvements (early stopping). The most important part is the "scenarios" options where we define all the scenarios we want to perform. First we define a scenario with no data augmentation where we override the "augment_data" option specifically for this scenario. Then we define augmentation scenarios where we change the values for data augmentation. Note that for the rotation option, we define a list of values, as is the case for the "flip" option. Mouffet will actually iterate over each of these lists and create all the permutations possible for these values. So we will actually train 4 models with different values for rotation and flip. To keep track of these values, we add a suffix to our model name. Each variable name between brackets will be replaced by the actual value of the variable during the training.

Training handler
----------------

Training in mouffet is handled by an instance of the class mouffet.training.TrainingHandler. This class handles logic related to the training options, creates training scenarios and for each of these scenarios, loads the right data, creates the appropriate model and then calls the train() method of the model with the data.  Each model is then saved, as well as the options that led to its creation. This ensure full traceability and reproductiblity.
To create an instance of this class, all that is required is the path to the training configuration file and the class of the mouffet.data.DataHandler used to manage the data. Once the object is created, all that remains is to call the train() method that will launch all training scenarios.
By default, Mouffet keeps a copy of each version of a model trained and statistics for this model if training was successful. By default, Mouffet will train a new version of each model each time the train() method of the training handler is called, but it is possible to avoid this by setting the "skip_trained" option to True.
For this example this gives:
CODE HERE
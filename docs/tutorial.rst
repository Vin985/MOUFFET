Tutorial: Creating image classifiers with Tensorflow
####################################################


Implementing a full example
===========================

To show the capabilities of Mouffet, we present here how to easily and robustly create several image classifiers using Mouffet and deep learning. To that end, we adapted the data augmentation tutorial found on the Tensorflow website (https://www.tensorflow.org/tutorials/images/data_augmentation) that aims to develop a simple deep learning model to classify images of flowers. Please note that this example aims only to showcase Mouffet features and not to be a deep learning tutorial nor a good example of model development. As Mouffet is independent of the implementation of the model, while we assume that users have a basic understanding of Python, it is not required to have any priori knowledge of working with Tensorflow or even deep learning.
Instead we will show which Python classes are required and how to use the configuration files to train and evaluate several models. We will demonstrate how by using Mouffet, we can easily investigate the effect of multiple values of data augmentation and data splitting.

Dependencies
============

Mouffet is written in Python v3.8 and requires the Python Standard Library (https://www.python.org). The package is lightweight and initially requires only three additional dependencies: "pandas" for dataframe manipulation (https://pandas.pydata.org), "pyyaml" for reading YAML configuration files (https://pypi.org/project/PyYAML/) and "feather-format" (https://pypi.org/project/feather-format/) to save dataframes in the feather format, a fast and lightweight saving format designed to be easily used with the R programming language.
Additionally, for the following example, the following packages will be required: "tensorflow", "tensorflow-datasets" and "sklearn".



Data management
===============

Data source
-----------

The data used in this example is taken from the"tf_flowers" database that can be downloaded and accessed via the tensorflow-datasets package. This database contains 3670 images of flowers separated among 5 classes (dandelion, daisy, tulips, sunflowers and roses). 
Data configuration
The data configuration file for this example is really simple. We define a database named "tf _flowers" in the "databases" options. Since the data will be downloaded from internet, there is no need to define paths here. We only define how the datasets will be split. Here we decide to put aside 80% of the dataset for training, 10% for validation and 10% for testing.

Data representation
-------------------

In Python, data is represented by two classes : mouffet.data.Database and mouffet.data.Dataset.
The mouffet.data.Dataset class describes the structure of the data and provides convenience functions for loading and saving datasets. Here we create a TFDataset class that inherits it.  The most important things to define are:
STRUCTURE:  A property that defines how the data is organized. When a dataset is loaded, it will create an object with this structure and fill it with the appropriate data. It is usually a simple python dict and its content are defined by the user. Here we create two entries "data" and "metadata" as this is how the data is provided when loading the "tf_flowers" dataset. 
load(): A method that actually loads the dataset. The main content of this function actually just calls the tensorflow_datasets.load() function with the database name and then fills a copy of the STRUCTURE object with the data loaded. 
talk about Data loaders? 
The mouffet.data.Database class stores information about a database, such as the paths where the data is located and provide functions to manage datasets such as checking if a dataset exists or splitting the data into the appropriate datasets. Here we create a TFExampleDatabase class that inherits. One of the most important thigs to change is the DATASET property that defines the class to use to create datasets. Here it is the TFDataset class we previously defined. In this example, we bypass the "check_dataset()" and "get_paths()" methods as these are irrelevant here since our dataset is dowladed from the internet and not present on our hard drive. 

Data manipulation
-----------------

In Mouffet, data is handled with an implementation of the mouffet.data.DataHandler class, called TFDatahandler here. This class mainly helps managing databases and datasets. One of its most important task is to prepare datasets training or evaluation, i.e. modify the raw data to fit the model.  Any modification to a dataset should be done in the prepare_dataset() method. This method takes two arguments, the dataset and "opts", a set of options for the current scenario that comes from the configuration file. We add in this function all the data augmentation features, as well as the shuffle and batch preparation logic for training and evaluation. Note that in this method, values for variables of interest are taken from the "opts" object via options we defined such as the angle of rotation via the "rotation" option or the way images should be flipped via the "flip" option.

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
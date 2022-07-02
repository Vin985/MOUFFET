Configuration files
===================


Data configuration file
-----------------------

Data configuration file: This is the file where all information related to data is described. This includes path to where the data is located but also how it is organized. Mouffet uses two levels to organize data, databases and datasets.
A database is a collection of data sharing a common theme. For example, it could be data coming from the same source or collected in the same location. If used for training a supervised machine learning model, annotations describing the objects the model will try to predict should also be provided.
A dataset is a subdivision of a database created for a specific model creation task. We identify three types of datasets (for more details, see our previous paper (Christin et al., 2021)):
Training dataset: used for training the model
Validation dataset: used for testing the model when the training is iterative. Usually a subset of the training dataset
Test dataset: used for evaluating the model. Should reflect the intended use case of the model

In its most simple expression, this files contains a list of databases under the "databases" option.
For each database, a "name" must be provided. Depending on how your database is organized, you can 
provide additional options such as the paths to each part of the database - e.g. the location of 
the training dataset, test dataset, annotations etc. - or how to split the datasets if needed.
Global options used for every database can be defined outside the "databases" option and overriden individually for each database.

Training configuration file
---------------------------

This is the file where all options related to the creation of a model are stored.
This includes the paths where to save the models, the options used to train a model but also the
data that will be used to train the model. As such, the training configuration file requires to be
linked to a data configuration file. Note that during this task, the training and validation 
datasets of a database will be used.

Evaluation configuration file
-----------------------------

This is the file containing all options related to the evaluation of a model. This includes the 
paths where to save the predictions and results of the evaluation, which model to use, and which
databases to use for the evaluation. As such, this file must also contain a reference to the data
configuration file. Note that this task uses the test datasets. These files also contain
information on which evaluators to use to assess the performance of the model and with what options
to use them (see below for more information on evaluators).


To avoid unnecessary duplication of the configuration files, it is also possible to define "parent" files that contain common options that do not change between iterations and will be inherited by the current files.
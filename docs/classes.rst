Core classes
============

Data representation
-------------------

In Python, data is represented by two classes : mouffet.data.Database and mouffet.data.Database.

Database
********

The 'Database' class stores information about a database, such as the paths where the data is
located and provide functions to manage datasets such as checking if a dataset exists or splitting
the data into the appropriate datasets.

Dataset
*******
The 'Dataset' class represents a dataset. It describes how the data is organized in Python, i.e. 
where to find raw data, annotations or metadata, for instance, to use during training or evaluation.
This class also provides convenience functions for loading and saving datasets.

DataLoader
**********

As more and more data sources become available, the same type of data can be stored in 
different ways on the hard drive. For instance, annotated images all represent the same type of
data: images and annotations. Thus instead of having to define several Dataset objects, Mouffet
provides mouffet.data.DataLoader classes whose sole purpose is to load data. The user can then
create a single AnnotatedImageDataset object that represent images and their annotations. And when
data needs to be loaded, different DataLoader classes can be used depending on the data sources.

Data handler
************

Data is handled with an implementation of the mouffet.data.DataHandler class. This class mainly helps
managing databases and datasets. One of its most important task is to prepare datasets for training
or evaluation, i.e. modify the raw data to fit the models. This could include data augmentation or
normalization, for instance by resizing images to make sure they all have the same size. Any
modification to a dataset should be done in the prepare_dataset() method which takes two arguments, 
the dataset and a set of options for the current scenario coming from the configuration file.
It is also possible to change the way data is prepared depending on the type of dataset, for
instance by defining a prepare_test_dataset() function that makes modifications only on test datasets.

Models
------

Models are at the core of the package. They are created during training and called during evaluation.
To define a model, users must provide an implementation of the mouffet.models.Model class.
The model classes are abstract and will not work without a custom implementation of some key methods
that vary depending on the type of model. Essentially, the user will have to define how to create,
load/save and use the model. As Mouffet was initially designed with deep learning in mind, mouffet
also provides the mouffet.model.DLModel class for this type of models.

Training handler
----------------

Training in mouffet is handled by an instance of the class mouffet.training.TrainingHandler.
This class handles logic related to the training options, creates training scenarios and for each
of these scenarios, loads the right data, creates the appropriate model and then calls the train()
method of the model with the appropriate data. Each model is then saved, as well as the options
that led to its creation. This ensure full traceability and reproductiblity.
To instantiate this class, only the path to the training configuration file and the class of the 
mouffet.data.DataHandler used to manage the data are required. Then all that remains is to call the
train() method which will launch all training scenarios. By default, Mouffet keeps a copy of each 
version of a model trained and statistics for this model if training was successful.
A new version of each model will be created each time the train() method of the training handler is
called unless otherwise specified.

Evaluation
----------

Evaluation handler
******************

Similarly to the training handler, the evaluation handler provides functions to load and expand evaluation scenarios from the evaluation configuration file. Then for each scenario it will load the required test dataset  load the model and call the model to predict the data. Note that only one dataset can be evaluated at the same time. To accelerate subsequent evaluations, all predictions will be saved and reused by default. An evaluator is then called (see below for detailed information). Once all scenarios have been run, all results are then compiled and saved with each set of options. It is possible to generate plots to describe all the results. They will then be compiled and saved at the end in a pdf format.

Evaluators
**********

This component takes a set of predictions made on a test dataset, compares them with the expected
results and then calculates evaluation metrics. Each evaluator should represent a different way of
evaluating the predictions. It is possible to generate plots at the evaluator. All plots for a similar
evaluators will then be compiled in a single multi-page pdf file. Mouffet also offers a basic
implementation for generating precision-recall curves using the plotnine package, an implementation
of the grammar of graphics in Python based on ggplot2 (link).
Here the value of using an evaluator even though most frameworks can provide one, is to be able to
save predictions and rerun all the evaluation independently while adding more flexibility in the
evaluation.
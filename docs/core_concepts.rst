Core concepts
=============


Databases
----------

At the core of any model creation is the data to be analyzed. When training a model, we define three
kind of data:
    - training data
    - validation data
    - test data

Mouffet allows the creation of data that fit into one of these categories and enforces that the
separation between all data is respected.

Databases are described in a yaml file which is described in further details here.


Models
------

versions
id

Evaluators
----------

    Detectors
    ---------

Scenarios
---------

- Training Scenarios


- Evaluation Scenarios

Data handlers
-------------

Datahandlers are classes that help handle the data.

Attributes
opts: options
tmp_db_data: a temporary object where loading data is saved. Could there be a better way?
databases: a dict with an association database_name : mouffet.options.DatabaseOptions object. Stores
the initial options for a given database. Each option object contains the global options overriden by
the specific options for the database

Data structures: the way data should be saved in memory. By default just a dict describing how the
information is saved.
Data types: by default pkl, feather if the key ends with "_df"

Paths: lists all path for a database using the provided information. Relevant paths are:

file_lists: list of paths of actual files used in the database

Splitting: functions to split the data should be provided.



Model handlers
--------------

- Trainer
- Evaluator

Detectors
---------




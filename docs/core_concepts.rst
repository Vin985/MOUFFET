Core concepts
=============

Databases
----------

Models
------

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




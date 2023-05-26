Frequently Asked Questions
==========================


All my audio files (training, validation and test) are at the same place. How do I do that?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the data_dir option when defining your database


I want to only use fine tuning and not train my whole model when performing transfer learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set n_epochs to 0 in the global model options.
In the 'fine_tuning' options, set a n_epoch > 0

I want to override a model option in a run during evaluation but it does not work when I add it
in 'model_options'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, a run will use the options used during training for evaluation. As in a run we do not
need to write the 'models' options, it becomes hard to override these values. The 'model_options'
category is mainly to provide default values to all the models. Therefore these values will be
overwritten if the same option is present in the training options. However we can go around this
by placing the desired option under a "scenario" option as these will then take precedence, if the
give scenario does not exist in the original training file.
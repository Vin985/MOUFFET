Training configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. csv-table::
    :header: "Option name", "Description", "Default", "Type"

    "databases_options", "Section containing default databases options related to this model. Overloads the information found in database config file. Can be overwritten in per mode scenarios", "{}", "Dict"
    "skip_similar", "Skip training of the model if another with the same options is found", False, "boolean"
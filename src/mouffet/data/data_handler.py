from ..utils import common_utils
from .database import Database


class DataHandler:
    """
    A class that handles all data related business. While this class provides convenience functions,
    this should be subclassed.


    .. csv-table::
        :header: "Option name", "Description", "Default", "Type"

        "generate_file_lists", "Should file lists be regenerated", False, "bool"
        "data_by_type", "Is the database split by type", False, "bool"

    """

    DATABASE_CLASS = Database

    def __init__(self, opts):
        self.opts = opts
        # self.tmp_db_data = None
        self.databases = self.load_databases()

    def load_databases(self):
        """Loads all databases defined in the 'databases' option of the configuration file.

        Returns:
            dict: A dict where keys are the names of the databases and values are instances
            of the DataHandler.OPTIONS_CLASS that must be a subclass of
            mouffet.options.DatabaseOptions
        """
        global_opts = dict(self.opts)
        databases = global_opts.pop("databases")

        databases = {
            database["name"]: self.DATABASE_CLASS(
                common_utils.deep_dict_update(dict(global_opts), database, copy=True)
            )
            for database in databases
        }
        return databases

    def duplicate_database(self, database):
        """Checks in the database list the database whose name is similar to the `database` argument.
        Then duplicates it and updates any options contained in `database`

        Args:
            database (instance of DataHandler.OPTIONS_CLASS): The database to duplicate

        Returns:
            mouffet.options.DatabaseOptions: The duplicated database
        """
        return self.DATABASE_CLASS(
            common_utils.deep_dict_update(
                self.databases[database["name"]].opts, database, copy=True
            ),
            database,
        )

    def update_database(self, new_opts=None, name="", copy=True):
        """Updates a database with the options contained in new_opts.
        If 'name' is not provided, this function tries to get the name of the database to update
        from the 'name' key in new_opts.

        Args:
            new_opts (dict, optional): A dictionary containing the new value to update.
            Defaults to None.
            name (str, optional): The name of the database to update. Defaults to "".
            copy (bool, optional): If True, returns a copy of the original database.
            Defaults to True.

        Raises:
            AttributeError: Thrown when no database 'name' has been found.

        Returns:
            DataHandler.OPTIONS_CLASS: An options object with the values of the original database
            with updated values. Returns None if the database name was not found.
        """
        new_opts = new_opts or {}
        name = name or new_opts.get("name", "")
        if not name:
            raise AttributeError(
                "A valid database name should be provided, either with the name option "
                + "or as a key in the new_opts dict"
            )
        if name in self.databases:
            return self.DATABASE_CLASS(
                common_utils.deep_dict_update(
                    self.databases[name].opts, new_opts, copy=copy
                ),
                new_opts,
            )
        return None

    def check_datasets(self, databases=None, db_types=None):
        databases = databases or self.databases.values()
        for database in databases:
            if isinstance(database, str):
                database = self.databases[database]
            database.check_database(db_types)

    def merge_datasets(self, datasets):
        merged = None
        for dataset in datasets.values():
            if not merged:
                merged = dataset.copy()
            for key in merged.data:
                if isinstance(dataset.data[key], list):
                    merged.data[key] += dataset.data[key]
                else:
                    merged.data[key].append(dataset.data[key])
        return merged

    def prepare_dataset(self, dataset, opts):
        """_summary_

        Args:
            dataset (_type_): _description_
            opts (_type_): _description_

        Returns:
            _type_: _description_
        """
        return dataset

    def get_database(self, name):
        return self.databases.get(name, None)

    def load_datasets(self, db_type, databases=None, by_dataset=False, **kwargs):
        """Load a dataset of type db_type.
        Can also prepare the dataset if the prepare argument is True.
        The user can provide a preparation function via prepare_func but by default will try to call
        a function named prepare_`db_type`_dataset (e.g. prepare_training_dataset) and then the
        generic prepare_dataset method.

        Args:
            db_type (_type_): _description_
            databases (_type_, optional): _description_. Defaults to None.
            by_dataset (bool, optional): _description_. Defaults to False.
            load_opts (_type_, optional): _description_. Defaults to None.
            prepare (bool, optional): _description_. Defaults to False.
            prepare_func (_type_, optional): _description_. Defaults to None.
            prepare_opts (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        res = {}
        databases = databases or self.databases.values()
        # * Iterate over databases
        for database in databases:
            # * Only load data if the give db_type is in the database definition
            if db_type in database.db_types:
                print(
                    "Loading {0} data for database: {1}".format(
                        db_type, database["name"]
                    )
                )
                res[database["name"]] = self.load_dataset(db_type, database, **kwargs)

        if not by_dataset:
            res = self.merge_datasets(res)

        return res

    def load_dataset(
        self,
        db_type,
        database,
        load_opts,
        prepare=False,
        prepare_func=None,
        prepare_opts=None,
    ):
        dataset = database.load_dataset(db_type, load_opts)
        if prepare:
            if prepare_func is None:
                db_func_name = "prepare_" + db_type + "_dataset"
                if hasattr(self, db_func_name):
                    prepare_func = getattr(self, db_func_name)
                else:
                    prepare_func = self.prepare_dataset
            dataset = prepare_func(dataset, prepare_opts)
        return dataset

    def get_summaries(self, db_types=None, databases=None, all=False, load_opts=None):
        res = {}
        databases = databases or self.databases.values()
        # * Iterate over databases
        for database in databases:
            ds_types = db_types or database.db_types
            # database.check_datasets(ds_types)

            # * Only load data if the give db_type is in the database definition
            for db_type in ds_types:
                if not db_type in database.db_types:
                    continue
                print(
                    "Generating summary for {0} data of database: {1}".format(
                        db_type, database["name"]
                    )
                )
                # try:
                dataset = database.load_dataset(db_type, load_opts)
                # except ValueError:
                # continue
                summary = dataset.summarize()

                if not database["name"] in res:
                    res[database["name"]] = {}

                res[database["name"]][db_type] = summary

        return res

from .options import Options


class DatabaseOptions(Options):

    DB_TYPES = ["test", "training", "validation"]

    DEFAULT_VALUES = {
        "class_type": "",
        "db_types": DB_TYPES,
        "data_extensions": [""],
        "generate_file_lists": False,
        "overwrite": False,
        "recursive": False,
        "save_intermediates": False,
        "tags_suffix": "-sceneRect.csv",
        "subfolders": None,
    }

    def has_type(self, db_type):
        return db_type in self.db_types

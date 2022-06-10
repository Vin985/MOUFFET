from .data_handler import DataHandler
from .data_loader import DataLoader
from .data_structure import DataStructure
from .dataset import Dataset
from .database import Database


DB_TYPE_TRAINING = "training"
DB_TYPE_VALIDATION = "validation"
DB_TYPE_TEST = "test"

ALL_DB_TYPES = [
    DB_TYPE_TRAINING,
    DB_TYPE_VALIDATION,
    DB_TYPE_TEST,
]

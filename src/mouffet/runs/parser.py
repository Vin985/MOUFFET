from argparse import ArgumentParser
from .utils import DEFAULTS


class RunArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            description="Perform training and evaluation runs in one go",
            **kwargs
        )
        self.add_argument(
            "runs", metavar="run", type=str, nargs="+", help="the name of the run"
        )

        self.add_argument(
            "-r",
            "--run_dir",
            default=DEFAULTS["run_dir"],
            help="The root directory where runs can be found",
        )

        self.add_argument(
            "-d",
            "--dest_dir",
            default=DEFAULTS["dest_dir"],
            help="The root directory where results will be saved",
        )

        self.add_argument(
            "-t",
            "--training_config",
            default=DEFAULTS["training_config"],
            help="The name of the training config files",
        )

        self.add_argument(
            "-e",
            "--evaluation_config",
            default=DEFAULTS["evaluation_config"],
            help="The name of the evaluation config files",
        )

        self.add_argument(
            "-D",
            "--data_config",
            default=DEFAULTS["data_config"],
            help="The name of the data config files",
        )

        self.add_argument(
            "-l",
            "--log_dir",
            default=DEFAULTS["log_dir"],
            help="The root directory where logs will be saved",
        )

        self.add_argument(
            "-c",
            "--clean",
            action="store_const",
            const=1,
            help="Clean logs and results directories for each run if they exist",
        )

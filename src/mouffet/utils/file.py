from pathlib import Path

import yaml

from . import common_utils
import csv


def ensure_path_exists(path, is_file=False):
    if isinstance(path, str):
        path = Path(path)
    if is_file:
        tmp = path.parent
    else:
        tmp = path
    if not tmp.exists():
        tmp.mkdir(exist_ok=True, parents=True)
    return path


def list_files(path, extensions=None, recursive=False):
    files = []
    extensions = extensions or []
    path = Path(path)
    if not path.exists():
        return files
    for item in path.iterdir():
        if item.is_dir() and recursive:
            files += list_files(item, extensions, recursive)
        elif item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return files


def list_folder(path, extensions=None):
    dirs = []
    files = []
    extensions = extensions or []
    for item in Path(path).iterdir():
        if item.is_dir():
            dirs.append(item)
        elif item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return (dirs, files)


def load_yaml(path):
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
        if config is None:
            config = {}
        return config


def load_config(opts_path, ignore_parent_path=False):
    opts = load_yaml(opts_path)
    if ignore_parent_path:
        return opts
    parent_path = opts.get("parent_path", "")
    if parent_path:
        parent = load_config(parent_path)
        opts = common_utils.deep_dict_update(parent, opts, copy=True)
    return opts


def get_full_path(path, root):
    path = Path(path)
    if path.is_absolute() or not root:
        return path
    else:
        return root / path


def load_csv_file(path):
    file_list = []
    with open(path, mode="r") as f:
        reader = csv.reader(f)
        for name in reader:
            file_list.append(Path(name[0]))
        print("Loaded file: " + str(path))
    return file_list


def load_file_lists(paths, db_types=None):
    res = {}
    for db_type, path in paths["file_list"].items():
        if db_types and db_type in db_types:
            file_list = load_csv_file(path)
            res[db_type] = file_list
    return res


def save_file_list(db_type, file_list, paths):
    file_list_path = paths["dest"][db_type] / (db_type + "_file_list.csv")
    with open(ensure_path_exists(file_list_path, is_file=True), mode="w") as f:
        writer = csv.writer(f)
        for name in file_list:
            writer.writerow([name])
        print("Saved file list:", str(file_list_path))

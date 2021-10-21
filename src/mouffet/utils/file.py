from pathlib import Path

import yaml

from . import common as common_utils


def ensure_path_exists(path, is_file=False):
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
        return config


def load_config(opts_path):
    opts = load_yaml(opts_path)
    parent_path = opts.get("parent_path", "")
    if parent_path:
        parent = load_config(parent_path)
        opts = common_utils.deep_dict_update(parent, opts, copy=True)
    return opts


def get_full_path(path, root):
    if path.is_absolute() or not root:
        return path
    else:
        return root / path

import random

from ..utils.file import list_folder


def random_split(path, split_props, extensions):

    splits = [[] for i in range(len(split_props) + 1)]

    files = [str(p) for p in path.rglob("*") if p.suffix.lower() in extensions]
    n_files = len(files)

    random.shuffle(files)
    idx = 0
    start_idx = 0

    for split_prop in split_props:
        n_split = int(split_prop * n_files)
        n_files = n_files - n_split
        splits[idx] = files[start_idx : start_idx + n_split]
        print(n_files, len(splits[idx]))
        start_idx = start_idx + n_split
        idx += 1
    splits[idx] = files[start_idx : len(files)]

    return splits


def split_list(data, split_props):
    splits = []
    tmp_list = data.copy()
    random.shuffle(tmp_list)
    for proportion in split_props:
        split_length = round(len(tmp_list) * proportion)
        splits.append(tmp_list[0:split_length])
        tmp_list = tmp_list[split_length:]
    splits.append(tmp_list)
    return splits


def split_folder(path, split_props, extensions):
    """Function if files are ordered with folders. Apply the split to each subfolder with files
    and continues recursively

    Args:
        path (Path): path where to start searching for files
        split_props (list): list of proportions for each split
        extensions (list): list of file extensions to include

    Returns:
        list: list with all the splits
    """
    splits = [[] for i in range(len(split_props) + 1)]
    dirs, files = list_folder(path, extensions)
    if files:
        tmp_splits = split_list(files, split_props)
        for i, split in enumerate(tmp_splits):
            splits[i] += split
    if dirs:
        for dir_path in dirs:
            tmp_splits = split_folder(dir_path, split_props, extensions)
            for i, split in enumerate(tmp_splits):
                splits[i] += split
    return splits

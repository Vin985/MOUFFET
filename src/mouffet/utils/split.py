import random

from .file import list_folder


def random_split(path, split_props, extensions):
    if not path.exists():
        raise ValueError(
            "'audio_dir' option must be provided to split into training and validation subsets"
        )
    files = [str(p) for p in path.rglob("*") if p.suffix.lower() in extensions]
    n_files = len(files)
    n_validation = int(split_props * n_files)
    validation_idx = random.sample(range(0, n_files), n_validation)
    training, validation = [], []
    for i in range(0, n_files):
        if i in validation_idx:
            validation.append(files[i])
        else:
            training.append(files[i])

    # Save file_list
    return {"training": training, "validation": validation}


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

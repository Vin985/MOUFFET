import collections.abc
from copy import deepcopy
from itertools import product


def deep_dict_update(original, update, copy=False, replace=True, except_keys=None):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    except_keys = except_keys or []
    if not isinstance(original, collections.Mapping):
        if copy:
            update = deepcopy(update)
        return update
    if copy:
        original = deepcopy(original)
    for key, value in update.items():
        if key in original and (
            (not replace and not key in except_keys) or (replace and key in except_keys)
        ):
            continue
        if isinstance(value, collections.Mapping):
            original[key] = deep_dict_update(original.get(key, {}), value, copy)
        else:
            original[key] = value
    return original


def listdict2dictlist(list_dict, flatten=False):
    """Function that takes a list of dict and converts it into a dict of lists

    Args:
        list_dict ([list]): The original list of dicts

    Returns:
        [dict]: A dict of lists
    """
    keys = {key for tmp_dict in list_dict for key in tmp_dict}
    res = {}
    for k in keys:
        tmp = []
        for d in list_dict:
            if k in d:
                val = d.get(k)
                if flatten and isinstance(val, list):
                    tmp += val
                else:
                    tmp.append(val)
        res[k] = tmp
    return res


def to_range(opts):
    return range_list(opts["start"], opts["end"], opts.get("step", 1))


def frange_positive(start, stop=None, step=None, endpoint=True, decimals=2):
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0

    count = 0
    has_end = False
    while True:
        temp = float(start + count * step)
        if temp >= stop:
            if endpoint:
                if has_end:
                    break
                else:
                    temp = stop
                    has_end = True
        if decimals:
            temp = round(temp, decimals)
        yield temp
        count += 1


def range_list(start, stop, step, endpoint=True, decimals=2):
    if isinstance(start, float) or isinstance(stop, float) or isinstance(step, float):
        return list(frange_positive(start, stop, step, endpoint, decimals))
    else:
        res = list(range(start, stop, step))
        if endpoint:
            res.append(stop)
        return res


def expand_options_dict(options):
    """Function to expand the options found in an options dict. If an option is a dict,
    two possibilities arise:
        - if the key "start" and "end" are present, then the dict is treated as a range and is
        replaced by a list with all values from the range. A "step" key can be found to define
        the step of the range. Default step: 1
        - Otherwise, the dict is expanded in a recursive manner.

    Args:
        options (dict): dict containing the options

    Returns:
        list: A list of dicts each containing a set of options
    """
    res = []
    tmp = []
    for val in options.values():
        if isinstance(val, dict):
            if "start" in val and "end" in val:
                tmp.append(to_range(val))
            else:
                tmp.append(expand_options_dict(val))
        else:
            if not isinstance(val, list):
                val = [val]
            tmp.append(val)
    for v in product(*tmp):
        d = dict(zip(options.keys(), v))
        res.append(d)
    return res


def get_dict_path(dict_obj, path, default=None, sep="--"):
    if not isinstance(path, list):
        path = path.split(sep)
    if not dict_obj:
        print("Warning! Empty dict provided. Returning default value")
        return default
    if not path:
        print("Warning! Empty path provided. Returning default value")
        return default

    key = path.pop(0)
    value = dict_obj.get(key, default)

    if path:
        if isinstance(value, dict):
            return get_dict_path(value, path, default)
        else:
            print(
                (
                    "Warning! Path does not exists as the value for key '{}' is not a dict."
                    + " Returning default value."
                ).format(key)
            )
            return default

    return value


def join_tuple(tuple, sep):
    tuple = list(filter(None, tuple))
    if len(tuple) > 1:
        return sep.join(tuple)
    return tuple[0]

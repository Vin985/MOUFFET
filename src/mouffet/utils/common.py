from collections.abc import Mapping
from copy import deepcopy
from itertools import product
import re


TERM_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "orange": "\033[33m",
    "blue": "\033[34m",
    "purple": "\033[35m",
    "cyan": "\033[36m",
    "lightgrey": "\033[37m",
    "darkgrey": "\033[90m",
    "lightred": "\033[91m",
    "lightgreen": "\033[92m",
    "yellow": "\033[93m",
    "lightblue": "\033[94m",
    "pink": "\033[95m",
    "lightcyan": "\033[96m",
    "reset": "\033[00m",
}


def print_color(msg, color):
    if not color in TERM_COLORS:
        raise ValueError("Color {} is not supported".format(color))
    print(TERM_COLORS[color] + msg + TERM_COLORS["reset"])


def print_error(msg):
    return print_color(msg, "red")


def print_warning(msg):
    return print_color(msg, "orange")


def print_title(msg):
    return print_color(msg, "lightgreen")


def print_info(msg):
    return print_color(msg, "yellow")


def deep_dict_update(original, update, copy=False, replace=True, except_keys=None):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    except_keys = except_keys or []
    if not isinstance(original, Mapping):
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
        if isinstance(value, Mapping):
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


def list2str(value, sep="-"):
    if isinstance(value, list):
        value = sep.join([list2str(i) for i in value])
    return str(value)


def resolve_dict_pattern(opts, pattern_name, path_separator="--"):
    pattern = opts.get(pattern_name, "")
    mid = ""
    if pattern:
        prefixes = opts.get(pattern_name + "_prepend", {})
        to_replace = re.findall("\\{(.+?)\\}", pattern)
        res = {}
        for key in to_replace:
            mid = ""
            if prefixes:
                prefix = prefixes.get(key, prefixes.get("default", ""))
                mid += str(prefix)
            if path_separator in key:
                value = get_dict_path(opts, key, key, sep=path_separator)
            else:
                value = opts.get(key, key)
            mid += list2str(value)
            res[key] = mid

        mid = pattern.format(**res)
    return mid


def any_in_list(elements, in_list):
    return len(set(elements).intersection(in_list)) > 0

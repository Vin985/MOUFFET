from mouffet.utils import common_utils


def test_dict_update():
    a = {"a": 1, "b": 2, "c": {"a": 1, "b": 2}}
    b = {"a": 3, "c": {"b": 3}}
    res = {"a": 3, "b": 2, "c": {"a": 1, "b": 3}}
    assert common_utils.deep_dict_update(a, b) == res


def test_dict_update_nocopy():
    a = {"a": 1, "b": 2, "c": {"a": 1, "b": 2}}
    b = {"a": 3, "c": {"b": 3}}
    tmp = common_utils.deep_dict_update(a, b)
    tmp["b"] = 4
    res = {"a": 3, "b": 4, "c": {"a": 1, "b": 3}}
    assert a == res


def test_dict_update_copy():
    a = {"a": 1, "b": 2, "c": {"a": 1, "b": 2}}
    b = {"a": 3, "c": {"b": 3}}
    tmp = common_utils.deep_dict_update(a, b, copy=True)
    tmp["b"] = 4
    res = {"a": 3, "b": 4, "c": {"a": 1, "b": 3}}
    assert a != res

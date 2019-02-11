def dict_recursive_bypass(dictionary: dict, on_node: callable) -> dict:
    """
    Recursive bypass dictionary

    :param dictionary:
    :param on_node: callable for every node, that get value of dict end node as parameters
    """
    res = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            res[k] = dict_recursive_bypass(v, on_node)
        else:
            res[k] = on_node(v)

    return res


def dict_pair_recursive_bypass(dictionary1: dict, dictionary2: dict, on_node: callable) -> dict:
    """
    Recursive bypass dictionary

    :param dictionary1:
    :param dictionary2:
    :param on_node: callable for every node, that get value of dict end node as parameters
    """
    res = {}
    for k, v in dictionary1.items():
        if isinstance(v, dict):
            res[k] = dict_pair_recursive_bypass(v, dictionary2[k], on_node)
        else:
            res[k] = on_node(v, dictionary2[k])

    return res

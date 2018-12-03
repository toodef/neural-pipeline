def dict_recursive_bypass(dictionary: dict, on_node: callable) -> dict:
    """
    Recursive bypass dictionary
    :param dictionary:
    :param on_node: callable for every node, that get key and value of dict end node as parameters
    """
    res = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            res[k] = dict_recursive_bypass(v, on_node)
        else:
            res[k] = on_node(k, v)

    return res

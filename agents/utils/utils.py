from typing import List


def clean_text(s, type_):
    assert type_ in ["inventory", "feedback", "description"]
    if type_ == "inventory":
        result = s.split("carrying")[1]
        if result[0] == ":":
            result = result[1:]
    elif type_ == "feedback":
        result = s
        if "$$$$$$$" in result:
            result = ""
        if "-=" in result:
            result = result.split("-=")[0]
    else:
        result = s

    result = result.strip()
    if not result:
        result = "nothing"
    return result


def idx_select(collection: List, indices: List, reversed_indices=False) -> List:
    """
    performs fancy indexing
    """
    if len(indices) == 0 or indices is None:
        return []
    if isinstance(indices[0], bool):
        if reversed_indices:
            indices = [not idx for idx in indices]
        return [collection[i] for i, idx in enumerate(indices) if idx]
    return [collection[idx] for idx in indices]

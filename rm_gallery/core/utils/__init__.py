"""
Core Utilities Module

Utility functions for RM-Gallery core.
"""


def get_value_by_path(data, path):
    """Get value from dictionary by path, supporting list indexing.

    Args:
        data (dict/list): The data dictionary or list to query
        path (str): Dot-separated path, e.g. "item.ticket_text" or "items.0.name"

    Returns:
        Any: The value at the path, or None if path doesn't exist
    """
    keys = path.split(".")
    current = data

    try:
        for key in keys:
            # Check if it's a list index (number)
            if isinstance(current, list) and key.isdigit():
                index = int(key)
                current = current[index]
            else:
                current = current[key]
        return current
    except (KeyError, TypeError, IndexError, ValueError):
        return None


def get_value_by_mapping(data, mapping) -> dict:
    """Get values from dictionary according to mapping.

    Args:
        data (dict): The data dictionary to query
        mapping (dict): Mapping relationship, key is path, value is field name

    Returns:
        dict: Mapped dictionary
    """
    result = {}
    for field, path in mapping.items():
        result[field] = get_value_by_path(data, path)
    return result


__all__ = [
    "get_value_by_mapping",
    "get_value_by_path",
]

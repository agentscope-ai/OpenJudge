import json
import os

import jsonlines
import yaml


def read_json(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist or is not a file.")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(data, file_path, ensure_ascii=False, indent=4):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=ensure_ascii, indent=indent)


def read_jsonl(file_path):
    """
    Load data from the json line.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist or is not a file.")

    content = []
    with jsonlines.open(file_path, mode="r") as reader:
        for obj in reader:
            content.append(obj)
    return content


def write_jsonl(file_path, data):
    """
    Write data to jsonl.
    """
    with jsonlines.open(file_path, mode="w") as writer:
        for item in data:
            writer.write(item)


def read_yaml(file_path):
    """
    Reads a YAML file and returns its content.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The content of the YAML file as a dictionary. Returns None if the file is not found or parsing fails.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    return None


def read_dataset(file_path: str):
    name, suffix = os.path.splitext(file_path)
    if suffix == ".json":
        return read_json(file_path)
    elif suffix == ".jsonl":
        return read_jsonl(file_path)
    elif suffix == ".yaml":
        return read_yaml(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

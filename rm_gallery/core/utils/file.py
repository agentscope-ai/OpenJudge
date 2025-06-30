import json
import os
import random
from copy import deepcopy
from typing import List

import jsonlines
import yaml

from rm_gallery.core.data.schema import DataSample


def read_json(file_path):
    """
    Reads JSON data from the specified file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist or is not a file.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist or is not a file.")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(data, file_path, ensure_ascii=False, indent=4):
    """
    Writes data to a JSON file.

    Args:
        data (Any): Data to be written to the JSON file.
        file_path (str): Path to the output JSON file.
        ensure_ascii (bool, optional): Whether to ensure ASCII encoding. Defaults to False.
        indent (int, optional): Indentation level for pretty-printing. Defaults to 4.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=ensure_ascii, indent=indent)


def read_jsonl(file_path):
    """
    Load data from the json line.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict]: List of JSON objects read from the file.

    Raises:
        FileNotFoundError: If the file does not exist or is not a file.
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

    Args:
        file_path (str): Path to the output JSONL file.
        data (List[Dict]): Data to be written to the JSONL file.
    """
    with jsonlines.open(file_path, mode="w") as writer:
        for item in data:
            writer.write(item)


def write_raw_content(file_path, data, auto_create_dir=True, mode="w"):
    """
    Writes raw text data to a file, optionally creating the directory path.

    Args:
        file_path (str): Path to the output file.
        data (List[str]): List of strings to be written line by line.
        auto_create_dir (bool, optional): Whether to automatically create the directory if it doesn't exist. Defaults to True.
    """
    dir_path = os.path.dirname(file_path)
    if auto_create_dir and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, mode) as f:
        for line in data:
            f.write(line)
            f.write("\n")


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
    """
    Reads dataset from a file based on its extension.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        Any: Dataset content parsed according to the file format.

    Raises:
        ValueError: If the file format is not supported.
    """
    name, suffix = os.path.splitext(file_path)
    if suffix == ".json":
        return read_json(file_path)
    elif suffix == ".jsonl":
        return read_jsonl(file_path)
    elif suffix == ".yaml":
        return read_yaml(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def split_samples(samples: List[dict | DataSample], ratio: float = 0.1):
    """
    Splits a list of samples into training and testing sets.

    Args:
        samples (List[Union[dict, DataSample]]): List of samples to split.
        ratio (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.1.

    Returns:
        Tuple[List[Union[dict, DataSample]], List[Union[dict, DataSample]]]: Train and test splits.
    """
    samples = deepcopy(samples)
    random.shuffle(samples)
    train_samples = samples[: int(len(samples) * ratio)]
    test_samples = samples[int(len(samples) * ratio) :]
    return train_samples, test_samples

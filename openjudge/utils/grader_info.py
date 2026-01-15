"""This module provides a utility function for collecting the core information of all Grader classes."""

import ast
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List

from loguru import logger


class _MethodInfo:
    def __init__(self, signature: str = "", docstring: str = ""):
        # covers name, arguments, and maybe return type
        self.signature = signature or ""
        self.docstring = docstring or ""

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


class _GraderInfo:
    """A class that stores core information of a Grader class"""

    def __init__(
        self,
        file_path: str = "",
        class_name: str = "",
        parent_class_names: list = None,
        init_method: _MethodInfo = None,
        aevaluate_method: _MethodInfo = None,
    ):
        self.file_path = file_path or ""
        self.class_name = class_name or ""
        self.parent_class_names = parent_class_names or []
        self.init_method = init_method or _MethodInfo()
        self.aevaluate_method = aevaluate_method or _MethodInfo()

    def __str__(self):
        d = deepcopy(self.__dict__)
        d["init_method"] = self.init_method.__dict__
        d["aevaluate_method"] = self.aevaluate_method.__dict__
        return json.dumps(d, ensure_ascii=False)

    def __iter__(self):
        """Customize dict(obj) result"""
        yield "file_path", self.file_path
        yield "class_name", self.class_name
        yield "parent_class_names", self.parent_class_names
        yield "init_method", self.init_method.__dict__
        yield "aevaluate_method", self.aevaluate_method.__dict__


def get_all_grader_info() -> List[Dict[str, Any]]:
    """Collect the information of all graders defined under the openjudge/graders folder."""
    t0 = time.time_ns()
    current_file_abs_path = Path(__file__).resolve()
    logger.info(f"grader_info.py path:{current_file_abs_path}")

    defs_of_classes_having_parent = {}
    grader_root_folder = Path(current_file_abs_path.parent.parent, "graders")
    logger.info(f"grader root folder:{grader_root_folder}")
    unprocessed_folders = Queue()
    unprocessed_folders.put(grader_root_folder)
    while not unprocessed_folders.empty():
        folder = unprocessed_folders.get()
        for item in folder.iterdir():
            if item.is_dir():
                unprocessed_folders.put(item)
            elif item.stem == "__init__" or item.suffix != ".py":
                # use safe heuristics to reduce candidate count
                continue
            else:
                _get_defs_of_classes_having_parent(item, defs_of_classes_having_parent)

    logger.info(f"classes having parent:{len(defs_of_classes_having_parent)}, {(time.time_ns() - t0)/1000000}ms")

    all_grader_class_defs = _get_grader_class_def(defs_of_classes_having_parent)

    t0 = time.time_ns()
    all_grader_info = []
    for _, (class_def, source_code) in all_grader_class_defs.items():
        grader_info = _parse_grader_class_def(class_def, source_code)
        all_grader_info.append(dict(grader_info))

    logger.info(f"all grader info:{len(all_grader_info)}, {(time.time_ns() - t0)/1000000}ms")

    return all_grader_info


def _get_defs_of_classes_having_parent(py_file: Path, defs_of_classes_having_parent: Dict[ast.ClassDef, str]):
    """Get the definitions and source codes of all classe that have parent class."""
    with open(py_file, "r", encoding="utf-8") as file:
        source_code = file.read()

    # Parse the source code into an AST node
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.bases:
            parent_count = len(node.bases)
            if parent_count > 1 or node.bases[0].id != "ABC":
                defs_of_classes_having_parent[node] = source_code


def _get_grader_class_def(defs_of_classes_having_parent: Dict[ast.ClassDef, str]):
    """Get the definitions of Grader classes"""
    t0 = time.time_ns()
    # the base grader class as seed
    all_grader_class_defs = {"BaseGrader": None}
    found_grader = True
    while found_grader:
        found_grader = False
        known_defs = []
        new_defs = []

        for class_def, source_code in defs_of_classes_having_parent.items():
            # Skip whose already processed
            if class_def.name in all_grader_class_defs:
                known_defs.append(class_def)
                continue

            for parent in class_def.bases:
                if parent.id in all_grader_class_defs:
                    all_grader_class_defs[class_def.name] = (class_def, source_code)
                    new_defs.append(class_def)
                    found_grader = True
                break

        # reduce inputs for the next round, by removing those processed in this round
        for n in known_defs:
            defs_of_classes_having_parent.pop(n)
        for n in new_defs:
            defs_of_classes_having_parent.pop(n)

    # remove the seed
    all_grader_class_defs.pop("BaseGrader")

    logger.info(f"all grader class defs:{len(all_grader_class_defs)}, {(time.time_ns() - t0)/1000000}ms")
    return all_grader_class_defs


_INIT_METHOD = "__init__"
_AEVALUATE_METHOD = "_aevaluate"
_TARGET_METHODS = set()
_TARGET_METHODS.add(_INIT_METHOD)
_TARGET_METHODS.add(_AEVALUATE_METHOD)

_NEWLINE_OR_MULTI_SPACE_PATTERN = re.compile(r"(\n|\s\s+)")


def _parse_grader_class_def(class_def: ast.ClassDef, source_code: str) -> _GraderInfo:
    """Use ast util to extract core information from a Grader class,
    and put the information into a _GraderInfo object."""

    init_method = _MethodInfo()
    aeval_method = _MethodInfo()
    # Find target methods within the class body
    for sub_node in class_def.body:
        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_name = sub_node.name
            if method_name not in _TARGET_METHODS:
                continue

            segment = ast.get_source_segment(source_code, sub_node)
            segment = _NEWLINE_OR_MULTI_SPACE_PATTERN.sub(" ", segment.strip())
            segment = segment.replace(") :", "):").replace(") ->", ")->").replace(" :", ":")

            # Method head ends in two ways: w/ or w/o return type annocation.
            # Figure it out.
            idx0 = segment.find("):")
            idx1 = segment.find(")->")
            if idx1 > 0:
                idx1 = segment.find(":", idx1)

            if idx0 > 0 and idx1 > 0:
                if idx0 < idx1:
                    idx = idx0 + 2
                else:
                    idx = idx1 + 1
            elif idx0 > 0:
                idx = idx0 + 2
            elif idx1 > 0:
                idx = idx1 + 1
            else:
                idx = -1

            # def foo(...) -> type:
            # def foo(...):
            if idx > 0:
                signature = segment[0:idx]
            else:
                signature = "SIGNATURE_NOT_FOUND"

            docstring = ast.get_docstring(sub_node)
            if method_name == _INIT_METHOD:
                init_method.signature = signature
                init_method.docstring = docstring
            elif method_name == _AEVALUATE_METHOD:
                aeval_method.signature = signature
                aeval_method.docstring = docstring

    grader_info_obj = _GraderInfo(
        class_name=class_def.name,
        parent_class_names=[p.id for p in class_def.bases],
        init_method=init_method,
        aevaluate_method=aeval_method,
    )

    return grader_info_obj


if __name__ == "__main__":
    graders = get_all_grader_info()
    print("-------------")
    print(f"{len(graders)} graders")
    for g_i in graders:
        print("-------------")
        print(type(g_i))
        print(g_i)

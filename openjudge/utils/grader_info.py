"""This module provides a utility function for collecting the core information of all Grader classes."""

import ast
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger


class MethodInfo:
    """Store critical information about a method."""

    def __init__(self, signature: str = "", docstring: str = ""):
        # covers name, arguments, and maybe return type
        self.signature = signature or ""
        self.docstring = docstring or ""

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


class GraderInfo:
    """A class that stores core information of a Grader class"""

    def __init__(
        self,
        module_path: str = "",
        class_name: str = "",
        parent_class_names: list = None,
        init_method: MethodInfo = None,
        aevaluate_method: MethodInfo = None,
    ):
        self.module_path = module_path or ""
        self.class_name = class_name or ""
        self.parent_class_names = parent_class_names or []
        self.init_method = init_method or None
        self.aevaluate_method = aevaluate_method or None

    def __str__(self):
        d = deepcopy(self.__dict__)
        d["init_method"] = self.init_method.__dict__
        d["aevaluate_method"] = self.aevaluate_method.__dict__
        return json.dumps(d, ensure_ascii=False)

    def __iter__(self):
        """Customize dict(obj) result"""
        yield "module_path", self.module_path
        yield "class_name", self.class_name
        yield "parent_class_names", self.parent_class_names
        yield "init_method", self.init_method.__dict__
        yield "aevaluate_method", self.aevaluate_method.__dict__


def get_all_grader_info() -> List[Dict[str, Any]]:
    """Collect the information of all graders defined under the openjudge/graders folder."""
    t0 = time.time_ns()
    current_file_abs_path = Path(__file__).resolve()
    logger.info(f"grader_info.py path:{current_file_abs_path}")

    # /path/to/OpenJudgeRepo
    project_root_folder = Path(current_file_abs_path.parent.parent.parent)
    logger.info(f"project root folder:{project_root_folder}")

    grader_root_folder = Path(current_file_abs_path.parent.parent, "graders")
    logger.info(f"grader root folder:{grader_root_folder}")

    defs_of_classes_having_parent = {}
    for f in grader_root_folder.rglob("**/*.py"):
        if f.stem != "__init__":
            _get_defs_of_classes_having_parent(f, project_root_folder, defs_of_classes_having_parent)

    logger.info(f"classes having parent:{len(defs_of_classes_having_parent)}, {(time.time_ns() - t0)/1000000}ms")

    all_grader_class_defs = _get_grader_class_def(defs_of_classes_having_parent)

    t0 = time.time_ns()
    all_grader_info = {}
    for _, (class_def, module_path_and_source_code_tuple) in all_grader_class_defs.items():
        grader_info = _parse_grader_class_def(class_def, module_path_and_source_code_tuple)
        all_grader_info[grader_info.class_name] = grader_info

    logger.info(f"all grader info:{len(all_grader_info)}, {(time.time_ns() - t0)/1000000}ms")

    return all_grader_info


def _get_defs_of_classes_having_parent(
    py_file: Path,
    open_judge_project_root_folder: Path,
    defs_of_classes_having_parent: Dict[ast.ClassDef, Tuple[str, str]],
):
    """Get the definitions, module paths, and source codes of all classe that have parent class."""
    # /path/to/OpenJudgeRepo/openjudge/graders/foo/my_grader.py ->
    # openjudge/graders/foo/my_grader.py ->
    # openjudge/graders/foo/my_grader ->
    # openjudge.graders.foo.my_grader
    module_path = str(py_file.relative_to(open_judge_project_root_folder).with_suffix("")).replace(os.sep, ".")

    with open(py_file, "r", encoding="utf-8") as file:
        source_code = file.read()

    # Parse the source code into an AST node
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.bases:
            parent_count = len(node.bases)
            if parent_count > 1 or node.bases[0].id != "ABC":
                defs_of_classes_having_parent[node] = (str(module_path), source_code)


def _get_grader_class_def(defs_of_classes_having_parent: Dict[ast.ClassDef, Tuple[str, str]]):
    """Get the definitions of Grader classes"""
    t0 = time.time_ns()
    # the base grader class as seed
    all_grader_class_defs = {"BaseGrader": None}
    found_grader = True
    while found_grader:
        found_grader = False
        known_defs = []
        new_defs = []

        for class_def, module_path_and_source_code_tuple in defs_of_classes_having_parent.items():
            # Skip whose already processed
            if class_def.name in all_grader_class_defs:
                known_defs.append(class_def)
                continue

            for parent in class_def.bases:
                if parent.id in all_grader_class_defs:
                    all_grader_class_defs[class_def.name] = (class_def, module_path_and_source_code_tuple)
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
_AEVALUATE_METHOD = "aevaluate"
_TARGET_METHODS = set()
_TARGET_METHODS.add(_INIT_METHOD)
_TARGET_METHODS.add(_AEVALUATE_METHOD)

_NEWLINE_OR_MULTI_SPACE_PATTERN = re.compile(r"(\n|\s\s+)")


def _parse_grader_class_def(class_def: ast.ClassDef, module_path_and_source_code_tuple: Tuple[str, str]) -> GraderInfo:
    """Use ast util to extract core information from a Grader class,
    and put the information into a GraderInfo object."""

    module_path = module_path_and_source_code_tuple[0]
    source_code = module_path_and_source_code_tuple[1]
    init_method = MethodInfo()
    aeval_method = MethodInfo()
    # Find target methods within the class body
    for sub_node in class_def.body:
        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_name = sub_node.name
            if method_name not in _TARGET_METHODS:
                continue

            segment = ast.get_source_segment(source_code, sub_node)
            # Find the colon ':' that ends the function signature by tracking parenthesis depth.
            open_parenthesis_count = 0
            end_of_func_signature_idx = -1
            for i, char in enumerate(segment):
                if char == "(":
                    open_parenthesis_count += 1
                elif char == ")":
                    open_parenthesis_count -= 1
                elif char == ":" and open_parenthesis_count == 0:
                    end_of_func_signature_idx = i
                    break

            if end_of_func_signature_idx > 0:
                # def foo(...): or def foo(...) -> bar:
                signature = segment[: end_of_func_signature_idx + 1]
                signature = _NEWLINE_OR_MULTI_SPACE_PATTERN.sub(" ", signature.strip())
                signature = signature.replace(" :", ":")
            else:
                signature = "SIGNATURE_NOT_FOUND"

            docstring = ast.get_docstring(sub_node)
            if method_name == _INIT_METHOD:
                init_method.signature = signature
                init_method.docstring = docstring
            elif method_name == _AEVALUATE_METHOD:
                aeval_method.signature = signature
                aeval_method.docstring = docstring

    grader_info_obj = GraderInfo(
        module_path=module_path,
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
    for name, g_i in graders.items():
        print("-------------")
        print(name)
        print(type(g_i))
        print(g_i)
        print(dict(g_i))

"""
Utility module for testing code solutions against test cases.

This module provides functionality for running code solutions against test cases
and evaluating their correctness. It supports both call-based and standard input
testing approaches with timeout protection and result comparison.
"""

import ast
import faulthandler
import json
import platform

# to run the solution files we're using a timing based approach
import signal
import sys
import traceback

# used for debugging to time steps
from datetime import datetime
from enum import Enum

# for capturing the stdout
from io import StringIO

# used for testing the code that reads from input
from typing import Any
from unittest.mock import mock_open, patch

import numpy as np
from loguru import logger
from pyext import RuntimeModule


def truncatefn(s, length=300):
    """
    Truncate a string to the specified length, placing ellipsis in the middle.

    Args:
        s (str): String to truncate
        length (int): Maximum length of the string. Defaults to 300.

    Returns:
        str: Truncated string with ellipsis in the middle if needed
    """
    assert isinstance(s, str)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    """
    Enum for code types.
    """

    call_based = 0
    standard_input = 1


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib


# pylint: disable=attribute-defined-outside-init
class Capturing(list):
    """
    Context manager to capture stdout output as a list.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def only_int_check(val):
    """
    Check if a value is an integer.

    Args:
        val: Value to check

    Returns:
        bool: True if value is an integer, False otherwise
    """
    return isinstance(val, int)


def string_int_check(val):
    """
    Check if a value is a string representation of an integer.

    Args:
        val: Value to check

    Returns:
        bool: True if value is a string containing only digits, False otherwise
    """
    return isinstance(val, str) and val.isdigit()


def combined_int_check(val):
    """
    Check if a value is either an integer or a string representation of an integer.

    Args:
        val: Value to check

    Returns:
        bool: True if value is an integer or string digit, False otherwise
    """
    return only_int_check(val) or string_int_check(val)


def clean_traceback(error_traceback):
    """
    Clean up traceback by removing unnecessary prefixes.

    Args:
        error_traceback (str): Original traceback string

    Returns:
        str: Cleaned traceback string
    """
    file_start = error_traceback.find('File "<string>"')
    logger.info(f"file_start = {file_start}")
    error_traceback = "Traceback (most recent call last):\n  " + error_traceback[file_start:]
    return error_traceback


# pylint: disable=too-many-statements, too-many-return-statements
def run_test(in_outs, test=None, timeout=15):
    """
    Run test cases against a generated code solution.

    Args:
        in_outs (dict): Dictionary containing inputs and expected outputs
        test (str, optional): Generated code solution to test
        timeout (int): Timeout in seconds for each test case

    Returns:
        tuple: (results, metadata) where results is a list of test results and
               metadata contains additional information about the test execution
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if not test:
        raise AssertionError("should not happen: missing test code input")

    method_name = None
    tmp = None
    which_type = None

    logger.debug(f"start = {datetime.now().time()}")

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    logger.debug(f"loaded input_output = {datetime.now().time()}")

    results = []
    sol = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""

    logger.debug(f"loading test code = {datetime.now().time()}")

    if which_type == CODE_TYPE.call_based:
        sol += test
        logger.debug(f"sol = {sol}")
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            error_traceback = traceback.format_exc()

            logger.debug(f"type 0 compilation error = {e}")
            results.append(-2)
            return results, {
                "error": repr(e),
                # "error_code": -1,
                # "error_message": "Compilation Error",
                "traceback": clean_traceback(error_traceback),
            }
        signal.alarm(0)

    elif which_type == CODE_TYPE.standard_input:
        # sol
        # if code has if __name__ == "__main__": then remove it
        try:
            astree = ast.parse(test)
            last_block = astree.body[-1]
            if isinstance(last_block, ast.If):
                condition = last_block.test
                if ast.unparse(condition).strip() == "__name__ == '__main__'":
                    test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
        except Exception:
            pass

        tmp_test = test.split("\n")

        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test += "def code():\n"
                new_test += i
                started = True
            elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                new_test += "\t" + i
            else:
                new_test += i
        tmp_test = new_test

        sol += tmp_test
        logger.debug(f"sol = {sol}")  # Debug statement removed
        method_name = "code"
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            error_traceback = traceback.format_exc()

            logger.debug(f"type 1 compilation error = {e}")
            results.append(-2)
            return results, {
                "error": repr(e),
                # "error_code": -1,
                # "error_message": "Compilation Error",
                "traceback": clean_traceback(error_traceback),
            }
        signal.alarm(0)

    logger.debug(f"get method = {datetime.now().time()}")

    try:
        method = getattr(
            tmp,
            method_name,
        )  # get_attr second arg must be str
    except Exception:
        signal.alarm(0)
        error_info = sys.exc_info()
        logger.debug(f"unable to get function error = {error_info}")
        results.append(-2)
        return results, {
            "error": repr(error_info),
            # "error_code": -1,
            # "error_message": "Unable to extract code",
            "traceback": clean_traceback(error_traceback),
        }

    for index, inputs in enumerate(in_outs["inputs"]):
        raw_inputs = inputs
        raw_outputs = in_outs["outputs"][index]
        if which_type == CODE_TYPE.call_based:
            inputs = [json.loads(line) for line in inputs.split("\n")]
            in_outs["outputs"][index] = json.loads(
                in_outs["outputs"][index],
            )

            truncate_line_size = 300 // (raw_inputs.count("\n") + 1)
            raw_inputs = "\n".join(
                [truncatefn(line, truncate_line_size) for line in raw_inputs.strip().split("\n")],
            )
            raw_outputs = truncatefn(raw_outputs, 200)
        else:
            raw_inputs = truncatefn(raw_inputs)
            raw_outputs = truncatefn(raw_outputs, 200)
        # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except Exception:
            pass
        try:
            if isinstance(in_outs["outputs"][index], dict):
                in_outs["outputs"][index] = [
                    {int(k): v for k, v in in_outs["outputs"][index].items()},
                ]
        except Exception:
            pass
        try:
            if isinstance(in_outs["outputs"][index][0], dict):
                in_outs["outputs"][index] = [
                    {int(k): v for k, v in in_outs["outputs"][index][0].items()},
                ]
        except Exception:
            pass
        logger.info(
            f"time: {datetime.now().time()}"
            f"testing index = {index}"
            f"inputs = {inputs}, {type(inputs)}. type = {which_type}",
        )
        if which_type == CODE_TYPE.call_based:  # Call-based
            signal.alarm(timeout)
            faulthandler.enable()
            try:
                output = method(*inputs)
                raw_true_output = output

                raw_true_output_copy = json.dumps(output)
                raw_true_output_copy = truncatefn(
                    raw_true_output_copy,
                    200,
                )

                # reference sequences are not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == in_outs["outputs"][index]
                if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                    tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                # reference sequences are not tuples
                try:
                    if isinstance(output[0], tuple):
                        tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                except Exception:
                    pass
                results.append(tmp_result)
                if tmp_result is not True:
                    return results, {
                        "output": raw_true_output_copy,
                        "expected": raw_outputs,
                        "inputs": raw_inputs,
                        # "error_code": -2,
                        "error_message": "Wrong Answer",
                    }
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                faulthandler.disable()

                logger.debug(f"Standard input runtime error or time limit exceeded error = {e}")
                results.append(-1)
                return results, {
                    "error": repr(e),
                    "traceback": clean_traceback(error_traceback),
                }
            faulthandler.disable()
            signal.alarm(0)
            logger.info(
                f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, "
                f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}",
            )
        elif which_type == CODE_TYPE.standard_input:  # Standard input
            faulthandler.enable()
            passed = False

            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(in_outs["outputs"][index], list):
                in_outs["outputs"][index] = "\n".join(
                    in_outs["outputs"][index],
                )

            signal.alarm(timeout)
            with Capturing() as output:
                try:
                    call_method(method, inputs)
                    # reset the alarm
                    signal.alarm(0)
                    passed = True
                except Exception as e:
                    # runtime error or took too long
                    signal.alarm(0)
                    error_traceback = traceback.format_exc()
                    logger.debug(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                    results.append(-1)
                    return results, {
                        "error": repr(e),
                        "traceback": clean_traceback(error_traceback),
                    }
                signal.alarm(0)
            raw_true_output = output[0]
            raw_true_output_copy = truncatefn(raw_true_output, 200)
            output = raw_true_output.splitlines()

            if not passed:
                nl = "\n"
                if not isinstance(inputs, list):
                    logger.info(
                        f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, "
                        f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                        f"{output == [in_outs['outputs'][index]]}",
                    )
                else:
                    logger.info(
                        f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, "
                        f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}",
                    )
                continue

            # reference sequences are expressed as lists not tuples
            if passed:
                logger.info(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

            if custom_compare_(output, in_outs["outputs"][index]):
                tmp_result = True
                results.append(tmp_result)
                continue

            # reference sequences are expressed as lists not tuples
            if isinstance(output, tuple):
                output = list(output)

            tmp_result = False
            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
                    if isinstance(output[0], str):
                        tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
            except Exception as e:
                logger.debug(f"Failed check1 exception = {e}")

            if tmp_result is True:
                results.append(tmp_result)
                continue

            # try one more time without \n
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = i.split("\n")
                    in_outs["outputs"][index][tmp_index] = [
                        x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                    ]
            else:
                in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                in_outs["outputs"][index] = list(
                    filter(len, in_outs["outputs"][index]),
                )
                in_outs["outputs"][index] = list(
                    map(lambda x: x.strip(), in_outs["outputs"][index]),
                )

            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])

            except Exception as e:

                logger.debug(f"Failed check2 exception = {e}")

            if tmp_result is True:
                results.append(tmp_result)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                output = list(filter(len, output))

            nl = "\n"
            if not isinstance(inputs, list):
                logger.info(
                    f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, "
                    f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                    f"{output == [in_outs['outputs'][index]]} {tmp_result=}",
                )
            else:
                logger.info(
                    f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, "
                    f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}",
                )

            logger.debug(f"{tmp_result=} @a")

            try:
                all_ints = all(
                    combined_int_check(e1) and combined_int_check(e2)
                    for e1, e2 in zip(output, in_outs["outputs"][index])
                )
                if not all_ints:
                    logger.debug(
                        [
                            combined_int_check(e1) and combined_int_check(e2)
                            for e1, e2 in zip(
                                output,
                                in_outs["outputs"][index],
                            )
                        ],
                    )
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                    )
            except Exception as e:
                logger.debug(f"Failed check3 exception = {e}")

            logger.debug(f"{tmp_result=} @b")

            try:
                if isinstance(output[0], list):
                    all_ints = all(
                        combined_int_check(e1) and combined_int_check(e2)
                        for e1, e2 in zip(
                            output[0],
                            in_outs["outputs"][index],
                        )
                    )
                    if not all_ints:
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                        )
            except Exception:
                pass

            if tmp_result is True:
                results.append(tmp_result)
                continue

            logger.debug(f"{tmp_result=} @d")
            # try by converting the stuff into split up list
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = set(i.split())
            else:
                in_outs["outputs"][index] = set(
                    in_outs["outputs"][index].split(),
                )

            logger.debug(f"{tmp_result=} @e")

            try:
                tmp_result = output == in_outs["outputs"][index]
            except Exception as e:
                logger.debug(f"Failed check4 exception = {e}")
                continue

            if tmp_result is True:
                results.append(tmp_result)
                continue

            logger.debug(f"{tmp_result=} @f")

            # try by converting the output into a split up list too
            if isinstance(output, list):
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = i.split()
                output = list(filter(len, output))
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = set(i)
            else:
                output = output.split()
                output = list(filter(len, output))
                output = set(output)

            logger.debug(f"{tmp_result=} @g")

            logger.debug("PASSED")

            results.append(tmp_result)
            if tmp_result is not True:
                return results, {
                    "output": raw_true_output_copy,
                    "expected": raw_outputs,
                    "inputs": raw_inputs,
                    # "error_code": -2,
                    "error_message": "Wrong Answer",
                }

            nl = "\n"
            if not isinstance(inputs, list):
                logger.info(
                    f"@2 output = {output}, test outputs = {in_outs['outputs'][index]},"
                    f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                    f"{output == [in_outs['outputs'][index]]}"
                )
            else:
                logger.info(
                    f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, "
                    f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                )

            logger.info(f"results = {results}")

    return results, {}


def custom_compare_(response, reference_response):
    """
    Custom comparison function for checking output against reference response.

    Args:
        response: Actual output from the tested code
        reference_response: Expected output

    Returns:
        bool: True if output matches reference response, False otherwise
    """
    if isinstance(response, list):
        output_1 = "\n".join(response)
        if stripped_string_compare(output_1, reference_response):
            return True

    if isinstance(response, list):
        output_2 = [o.lstrip().rstrip() for o in response]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, reference_response):
            return True

    return False


def stripped_string_compare(s1, s2):
    """
    Compare two strings after stripping whitespace.

    Args:
        s1 (str): First string
        s2 (str): Second string

    Returns:
        bool: True if stripped strings are equal, False otherwise
    """
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    """
    Call a method with mocked input sources.

    Args:
        method: Method to call
        inputs: Input data to provide to the method

    Returns:
        Result of calling the method
    """
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    # pylint: disable=inconsistent-return-statements
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            return

    return _inner_call_method(method)


# pylint: disable=too-many-statements
def reliability_guard(maximum_memory_bytes: Any | None = None):
    """
    Disable various destructive functions to prevent generated code from interfering with testing.

    This function is NOT a security sandbox. Untrusted code should not be blindly executed.

    Args:
        maximum_memory_bytes: Maximum memory limit in bytes
    """
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (maximum_memory_bytes, maximum_memory_bytes),
        )
        resource.setrlimit(
            resource.RLIMIT_DATA,
            (maximum_memory_bytes, maximum_memory_bytes),
        )
        if platform.uname().system != "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None  # 防止干扰repl评测
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

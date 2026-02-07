# -*- coding: utf-8 -*-
"""
Prompt Format Checker

Validates that prompts follow the standardized XML tag format.

Format Structure:
1. Objective (任务目标) - Required: Opening statement describing the evaluator's role
2. Evaluation (评估过程):
   2.1 Rubrics (评分标准) - Required
   2.2 Steps (评估步骤) - Optional
   2.3 Constraints (注意事项) - Optional
   2.4 Scale (评分量表) - Required
3. Data (数据) - Required: Input data with XML tags (just check pairs)
4. Examples (参考示例) - Optional
5. Schema (输出格式) - Required

Usage:
    # Check a single grader file
    python prompt_format_checker.py path/to/grader.py

    # Check all graders in a directory
    python prompt_format_checker.py path/to/graders/

    # Check with strict mode
    python prompt_format_checker.py path/to/graders/ --strict
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Language(Enum):
    """Language enum for prompt validation."""

    EN = "en"
    ZH = "zh"
    MIXED = "mixed"


@dataclass
class TagDefinition:
    """Definition of a required or optional XML tag."""

    en_tag: str
    zh_tag: str
    required: bool = True
    description: str = ""


@dataclass
class ValidationResult:
    """Result of prompt format validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    found_tags: Dict[str, bool] = field(default_factory=dict)
    language: Optional[Language] = None
    source: str = ""  # Source file or variable name

    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = [f"Validation Result: {status}"]

        if self.source:
            result.append(f"Source: {self.source}")

        if self.language:
            result.append(f"Detected Language: {self.language.value}")

        if self.errors:
            result.append("\nErrors:")
            for error in self.errors:
                result.append(f"  ❌ {error}")

        if self.warnings:
            result.append("\nWarnings:")
            for warning in self.warnings:
                result.append(f"  ⚠️ {warning}")

        if self.found_tags:
            result.append("\nFound Tags:")
            for tag, found in self.found_tags.items():
                status_icon = "✓" if found else "✗"
                result.append(f"  {status_icon} {tag}")

        return "\n".join(result)


class PromptFormatChecker:
    """
    Validates prompt format against the standardized XML tag structure.

    The checker validates:
    1. Required tags are present
    2. Tags are properly opened and closed
    3. Tag naming follows conventions (EN or ZH)

    Example:
        >>> checker = PromptFormatChecker()
        >>> result = checker.validate(prompt_text)
        >>> print(result)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    # Define all tags with their English and Chinese versions
    TAGS: List[TagDefinition] = [
        # Section 2.1: Rubrics (Required)
        TagDefinition(
            en_tag="Rubrics",
            zh_tag="评分标准",
            required=True,
            description="Evaluation criteria and standards",
        ),
        # Section 2.2: Steps (Optional)
        TagDefinition(
            en_tag="Steps",
            zh_tag="评估步骤",
            required=False,
            description="Step-by-step evaluation process",
        ),
        # Section 2.3: Constraints (Optional)
        TagDefinition(
            en_tag="Constraints",
            zh_tag="注意事项",
            required=False,
            description="Constraints and considerations",
        ),
        # Section 2.4: Scale (Required)
        TagDefinition(
            en_tag="Scale",
            zh_tag="评分量表",
            required=True,
            description="Scoring scale definitions",
        ),
        # Section 4: Examples (Optional)
        TagDefinition(
            en_tag="Examples",
            zh_tag="参考示例",
            required=False,
            description="Example evaluations",
        ),
        # Section 5: Output Schema (Required)
        TagDefinition(
            en_tag="Output Schema",
            zh_tag="输出格式",
            required=True,
            description="Expected output format",
        ),
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the checker.

        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode

    def _detect_language(self, prompt: str) -> Language:
        """Detect the primary language of the prompt based on tags used."""
        en_count = 0
        zh_count = 0

        for tag_def in self.TAGS:
            if f"<{tag_def.en_tag}>" in prompt or f"</{tag_def.en_tag}>" in prompt:
                en_count += 1
            if f"<{tag_def.zh_tag}>" in prompt or f"</{tag_def.zh_tag}>" in prompt:
                zh_count += 1

        if en_count > 0 and zh_count > 0:
            return Language.MIXED
        elif zh_count > en_count:
            return Language.ZH
        else:
            return Language.EN

    def _check_tag_pair(self, prompt: str, tag_name: str) -> Tuple[bool, bool, bool]:
        """
        Check if a tag is properly opened and closed.

        Handles tags with (Optional) suffix: <Context (Optional)> pairs with </Context>

        Returns:
            Tuple of (has_open, has_close, is_valid)
        """
        open_tag = f"<{tag_name}>"

        # Remove (Optional) or （可选） suffix for closing tag
        base_tag_name = re.sub(r"\s*\(Optional\)\s*$", "", tag_name, flags=re.IGNORECASE)
        base_tag_name = re.sub(r"\s*（可选）\s*$", "", base_tag_name).strip()
        close_tag = f"</{base_tag_name}>"

        has_open = open_tag in prompt
        has_close = close_tag in prompt

        if has_open and has_close:
            # Check that close comes after open
            open_pos = prompt.find(open_tag)
            close_pos = prompt.find(close_tag)
            is_valid = close_pos > open_pos
        else:
            is_valid = False

        return has_open, has_close, is_valid

    def _find_all_tags(self, prompt: str) -> List[str]:
        """Find all XML-style tags in the prompt."""
        # Match both opening and closing tags
        pattern = r"</?([^>]+)>"
        matches = re.findall(pattern, prompt)
        return list(set(matches))

    def _check_all_tag_pairs(self, prompt: str) -> List[str]:
        """
        Check that all tags are properly paired (opened and closed).

        Returns:
            List of unpaired tags
        """
        unpaired = []

        # Patterns to ignore (JSON placeholders, not real XML tags)
        # These are typically found in output schema examples like:
        # "score": <0.0 or 1.0>, "reason": "<detailed explanation>"
        ignore_patterns = [
            r"^[\d\.\s]",  # Starts with numbers (e.g., "0.0 or 1.0")
            r"^int",  # Type hints like "int or float"
            r"^float",
            r"^str",
            r"详细",  # Chinese placeholders
            r"detailed",  # English placeholders
            r"explanation",
            r"整数",
            r"浮点",
        ]

        def should_ignore(tag: str) -> bool:
            """Check if a tag should be ignored (likely a JSON placeholder)."""
            for pattern in ignore_patterns:
                if re.search(pattern, tag, re.IGNORECASE):
                    return True
            return False

        def get_base_tag_name(tag: str) -> str:
            """
            Extract base tag name, removing (Optional) or （可选） suffixes.
            e.g., "Context (Optional)" -> "Context"
            """
            # Remove (Optional) or （可选） suffix
            tag = re.sub(r"\s*\(Optional\)\s*$", "", tag, flags=re.IGNORECASE)
            tag = re.sub(r"\s*（可选）\s*$", "", tag)
            return tag.strip()

        # Find all opening tags
        open_pattern = r"<([^/][^>]*)>"
        open_tags = re.findall(open_pattern, prompt)

        # Find all closing tags
        close_pattern = r"</([^>]+)>"
        close_tags = re.findall(close_pattern, prompt)

        # Build a set of base names from closing tags for matching
        close_tag_bases = {get_base_tag_name(tag) for tag in close_tags}

        # Check for unclosed opening tags
        for tag in set(open_tags):
            if should_ignore(tag):
                continue
            base_name = get_base_tag_name(tag)
            # Check if there's a matching closing tag (by base name)
            if base_name not in close_tag_bases:
                unpaired.append(f"<{tag}> (unclosed)")

        # Build a set of base names from opening tags for matching
        open_tag_bases = {get_base_tag_name(tag) for tag in open_tags}

        # Check for closing tags without opening tags
        for tag in set(close_tags):
            if should_ignore(tag):
                continue
            base_name = get_base_tag_name(tag)
            # Check if there's a matching opening tag (by base name)
            if base_name not in open_tag_bases:
                unpaired.append(f"</{tag}> (no opening tag)")

        return unpaired

    def validate(self, prompt: str, source: str = "") -> ValidationResult:
        """
        Validate a prompt against the format specification.

        Args:
            prompt: The prompt text to validate
            source: Optional source identifier (file path or variable name)

        Returns:
            ValidationResult with validation status, errors, and warnings
        """
        errors: List[str] = []
        warnings: List[str] = []
        found_tags: Dict[str, bool] = {}

        # Detect language
        language = self._detect_language(prompt)

        # Check required and optional tags
        for tag_def in self.TAGS:
            # Determine which tag name to check based on language
            if language == Language.ZH:
                primary_tag = tag_def.zh_tag
                alt_tag = tag_def.en_tag
            else:
                primary_tag = tag_def.en_tag
                alt_tag = tag_def.zh_tag

            # Check primary tag
            has_open, has_close, is_valid = self._check_tag_pair(prompt, primary_tag)

            # If not found, check alternative tag
            if not has_open:
                alt_open, alt_close, alt_valid = self._check_tag_pair(prompt, alt_tag)
                if alt_open:
                    has_open, has_close, is_valid = alt_open, alt_close, alt_valid
                    primary_tag = alt_tag

            tag_display = f"{tag_def.en_tag}/{tag_def.zh_tag}"
            found_tags[tag_display] = is_valid

            if tag_def.required:
                if not has_open:
                    errors.append(f"Missing required tag: <{tag_display}>")
                elif not has_close:
                    errors.append(f"Tag <{primary_tag}> is not properly closed")
                elif not is_valid:
                    errors.append(f"Tag <{primary_tag}> has invalid structure (close before open)")
            else:
                # Optional tag
                if has_open and not has_close:
                    warnings.append(f"Optional tag <{primary_tag}> is not properly closed")
                elif has_open and not is_valid:
                    warnings.append(f"Optional tag <{primary_tag}> has invalid structure")

        # Check all tag pairs (simple validation: just ensure <> and </> match)
        unpaired_tags = self._check_all_tag_pairs(prompt)
        for unpaired in unpaired_tags:
            # Skip if already reported in required tags
            tag_name = unpaired.split()[0].strip("<>/")
            already_reported = any(tag_name in err for err in errors)
            if not already_reported:
                warnings.append(f"Unpaired tag: {unpaired}")

        # Check for data tags (Required: at least one data input tag must be present)
        # Find all tags in the prompt that are not in the predefined TAGS list
        all_found_tags = self._find_all_tags(prompt)
        predefined_tag_names = set()
        for tag_def in self.TAGS:
            predefined_tag_names.add(tag_def.en_tag)
            predefined_tag_names.add(tag_def.zh_tag)

        # Data tags are any properly paired tags that are not predefined evaluation tags
        data_tags = []
        for tag in all_found_tags:
            if tag not in predefined_tag_names:
                _, _, is_valid = self._check_tag_pair(prompt, tag)
                if is_valid:
                    data_tags.append(tag)

        if not data_tags:
            errors.append(
                "Missing required data tags: At least one data input tag is required "
                "(e.g., <Query>, <Response>, <Context>)"
            )

        # Check for JSON output indicator
        if "JSON:" not in prompt and "json" not in prompt.lower():
            warnings.append("Missing 'JSON:' indicator at the end of the prompt")

        # Determine validity
        if self.strict_mode:
            is_valid = len(errors) == 0 and len(warnings) == 0
        else:
            is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            found_tags=found_tags,
            language=language,
            source=source,
        )

    def _extract_string_from_call(self, call_node: ast.Call) -> Optional[str]:
        """
        Recursively extract string from a Call node.

        Handles patterns like:
        - textwrap.dedent("...")
        - textwrap.dedent("...").strip()
        - "...".strip()
        """
        # Check if it's a method call like .strip()
        if isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr

            # If it's .strip() or similar, look at the object it's called on
            if method_name in ("strip", "lstrip", "rstrip"):
                obj = call_node.func.value
                if isinstance(obj, ast.Constant):
                    return obj.value
                elif isinstance(obj, ast.Call):
                    return self._extract_string_from_call(obj)

            # Check for textwrap.dedent()
            if method_name == "dedent":
                if call_node.args and isinstance(call_node.args[0], ast.Constant):
                    return call_node.args[0].value

        # Check for direct function call like dedent()
        if isinstance(call_node.func, ast.Name):
            if call_node.func.id == "dedent":
                if call_node.args and isinstance(call_node.args[0], ast.Constant):
                    return call_node.args[0].value

        return None

    def extract_prompts_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Extract prompt strings from a Python grader file.

        Looks for variables ending with _PROMPT_EN, _PROMPT_ZH, or similar patterns.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary mapping variable names to prompt strings
        """
        prompts = {}

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"  ⚠️ Syntax error in {file_path}: {e}")
            return prompts

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        # Look for prompt variable patterns
                        if "PROMPT" in var_name.upper():
                            # Try to extract the string value
                            try:
                                if isinstance(node.value, ast.Constant):
                                    prompts[var_name] = node.value.value
                                elif isinstance(node.value, ast.Call):
                                    # Handle various call patterns
                                    extracted = self._extract_string_from_call(node.value)
                                    if extracted:
                                        prompts[var_name] = extracted
                            except Exception:
                                pass

        return prompts

    def validate_file(self, file_path: str) -> List[ValidationResult]:
        """
        Validate all prompts in a Python grader file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of ValidationResult for each prompt found
        """
        results = []
        prompts = self.extract_prompts_from_file(file_path)

        if not prompts:
            # Return empty result indicating no prompts found
            return [
                ValidationResult(
                    is_valid=True,
                    warnings=["No prompt variables found in file"],
                    source=file_path,
                )
            ]

        for var_name, prompt in prompts.items():
            source = f"{file_path}::{var_name}"
            result = self.validate(prompt, source=source)
            results.append(result)

        return results

    def validate_directory(self, dir_path: str, pattern: str = "*.py") -> List[ValidationResult]:
        """
        Validate all grader files in a directory.

        Args:
            dir_path: Path to the directory
            pattern: Glob pattern for matching files (default: *grader*.py)

        Returns:
            List of ValidationResult for all prompts found
        """
        results = []
        path = Path(dir_path)

        # Find all matching Python files recursively
        for py_file in path.rglob(pattern):
            if py_file.is_file():
                file_results = self.validate_file(str(py_file))
                results.extend(file_results)

        return results


def check_prompt_format(prompt: str, strict: bool = False) -> ValidationResult:
    """
    Convenience function to check prompt format.

    Args:
        prompt: The prompt text to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with validation status

    Example:
        >>> result = check_prompt_format(my_prompt)
        >>> if not result.is_valid:
        ...     print("Prompt format is invalid!")
        ...     for error in result.errors:
        ...         print(f"  - {error}")
    """
    checker = PromptFormatChecker(strict_mode=strict)
    return checker.validate(prompt)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate prompt format in grader files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check a single grader file
    python prompt_format_checker.py path/to/grader.py

    # Check all graders in a directory
    python prompt_format_checker.py path/to/graders/

    # Check with strict mode (warnings become errors)
    python prompt_format_checker.py path/to/graders/ --strict

    # Custom file pattern
    python prompt_format_checker.py path/to/graders/ --pattern "*.py"
        """,
    )
    parser.add_argument("path", help="Path to a Python file or directory to validate")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="File pattern for directory search (default: *.py)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show errors and summary")

    args = parser.parse_args()

    checker = PromptFormatChecker(strict_mode=args.strict)
    path = Path(args.path)

    if not path.exists():
        print(f"❌ Path does not exist: {args.path}")
        return 1

    results = []

    if path.is_file():
        if not args.quiet:
            print(f"Checking file: {path}")
        results = checker.validate_file(str(path))
    elif path.is_dir():
        if not args.quiet:
            print(f"Checking directory: {path}")
            print(f"Pattern: {args.pattern}")
        results = checker.validate_directory(str(path), pattern=args.pattern)
    else:
        print(f"❌ Invalid path: {args.path}")
        return 1

    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    valid_count = 0
    invalid_count = 0
    invalid_results = []

    # First pass: print valid results, collect invalid ones
    for result in results:
        if result.is_valid:
            valid_count += 1
            if not args.quiet:
                print(f"\n{result}")
        else:
            invalid_count += 1
            invalid_results.append(result)

    # Print invalid results at the end
    if invalid_results:
        print("\n" + "-" * 60)
        print("Invalid Prompts")
        print("-" * 60)
        for result in invalid_results:
            print(f"\n{result}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total prompts checked: {len(results)}")
    print(f"  ✅ Valid: {valid_count}")
    print(f"  ❌ Invalid: {invalid_count}")

    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

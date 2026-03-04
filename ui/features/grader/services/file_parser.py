# -*- coding: utf-8 -*-
"""File parser for batch evaluation data.

Supports JSON and CSV formats with validation against grader input requirements.
"""

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from loguru import logger

# Maximum number of records allowed per batch
MAX_BATCH_SIZE = 5000

# Standard input fields for different grader types
STANDARD_FIELDS = ["query", "response", "reference_response", "context"]
AGENT_FIELDS = ["query", "tool_definitions", "tool_calls", "reference_tool_calls"]
MULTIMODAL_FIELDS = ["response_multimodal", "response_image"]


@dataclass
class ParseResult:
    """Result of file parsing operation."""

    success: bool
    data: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record_count: int = 0
    file_format: str = ""


@dataclass
class ValidationResult:
    """Result of data validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_file_format(filename: str) -> str:
    """Detect file format from filename extension.

    Args:
        filename: Name of the file

    Returns:
        'json' or 'csv' or 'unknown'
    """
    lower_name = filename.lower()
    if lower_name.endswith(".json"):
        return "json"
    elif lower_name.endswith(".csv"):
        return "csv"
    return "unknown"


def _decode_content(content: bytes) -> str:
    """Decode file content with encoding detection.

    Handles UTF-8 BOM and falls back to common encodings.

    Args:
        content: Raw file content bytes

    Returns:
        Decoded string content
    """
    # Try UTF-8 with BOM
    if content.startswith(b"\xef\xbb\xbf"):
        return content[3:].decode("utf-8")

    # Try UTF-8
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Try GBK (common for Chinese CSV)
    try:
        return content.decode("gbk")
    except UnicodeDecodeError:
        pass

    # Try Latin-1 as fallback (accepts any byte)
    return content.decode("latin-1")


def parse_json_file(content: bytes) -> ParseResult:
    """Parse JSON file content.

    Expects either:
    - {"data": [...]} format
    - Direct array [...]

    Args:
        content: Raw file content bytes

    Returns:
        ParseResult with parsed data or errors
    """
    result = ParseResult(success=False, file_format="json")

    try:
        text = _decode_content(content)
        parsed = json.loads(text)

        # Handle both formats
        if isinstance(parsed, dict):
            if "data" in parsed:
                data = parsed["data"]
            else:
                # Assume the dict itself is a single record
                data = [parsed]
        elif isinstance(parsed, list):
            data = parsed
        else:
            result.errors.append("JSON must be an array or object with 'data' field")
            return result

        if not isinstance(data, list):
            result.errors.append("Data must be an array of records")
            return result

        # Check max size
        if len(data) > MAX_BATCH_SIZE:
            result.errors.append(f"Too many records: {len(data)}. Maximum allowed is {MAX_BATCH_SIZE}")
            return result

        if len(data) == 0:
            result.errors.append("Data array is empty")
            return result

        # Validate each record is a dict
        valid_records = []
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                result.warnings.append(f"Record {i + 1} is not an object, skipped")
                continue
            valid_records.append(record)

        if not valid_records:
            result.errors.append("No valid records found in data")
            return result

        result.success = True
        result.data = valid_records
        result.record_count = len(valid_records)

    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.exception("Error parsing JSON file")
        result.errors.append(f"Error parsing file: {e}")

    return result


def parse_csv_file(content: bytes) -> ParseResult:
    """Parse CSV file content.

    Args:
        content: Raw file content bytes

    Returns:
        ParseResult with parsed data or errors
    """
    result = ParseResult(success=False, file_format="csv")

    try:
        text = _decode_content(content)

        # Detect dialect
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except csv.Error:
            dialect = csv.excel  # Default to excel dialect

        reader = csv.DictReader(io.StringIO(text), dialect=dialect)

        if not reader.fieldnames:
            result.errors.append("CSV file has no headers")
            return result

        # Normalize header names (strip whitespace, lowercase for comparison)
        _ = {h.strip().lower(): h.strip() for h in reader.fieldnames}

        records = []
        for i, row in enumerate(reader):
            if i >= MAX_BATCH_SIZE:
                result.warnings.append(
                    f"File has more than {MAX_BATCH_SIZE} records. " f"Only first {MAX_BATCH_SIZE} will be processed."
                )
                break

            # Clean up record: strip whitespace from keys and values
            record = {}
            for key, value in row.items():
                if key:
                    clean_key = key.strip()
                    clean_value = value.strip() if value else ""
                    # Handle empty strings as None for optional fields
                    record[clean_key] = clean_value if clean_value else None

            records.append(record)

        if not records:
            result.errors.append("CSV file has no data rows")
            return result

        result.success = True
        result.data = records
        result.record_count = len(records)

    except Exception as e:
        logger.exception("Error parsing CSV file")
        result.errors.append(f"Error parsing CSV file: {e}")

    return result


def parse_file(file: BinaryIO, filename: str) -> ParseResult:
    """Parse uploaded file based on format.

    Args:
        file: File-like object with read() method
        filename: Name of the file for format detection

    Returns:
        ParseResult with parsed data or errors
    """
    file_format = detect_file_format(filename)

    if file_format == "unknown":
        return ParseResult(
            success=False,
            errors=["Unsupported file format. Please upload JSON or CSV file."],
        )

    content = file.read()

    if not content:
        return ParseResult(success=False, errors=["File is empty"])

    if file_format == "json":
        return parse_json_file(content)
    else:
        return parse_csv_file(content)


def validate_data_for_grader(
    data: list[dict[str, Any]],
    grader_config: dict[str, Any],
) -> ValidationResult:
    """Validate parsed data against grader requirements.

    Args:
        data: List of parsed records
        grader_config: Grader configuration from registry

    Returns:
        ValidationResult with validation status and any issues
    """
    result = ValidationResult(valid=True)

    if not data:
        result.valid = False
        result.errors.append("No data to validate")
        return result

    input_fields = grader_config.get("input_fields", ["query", "response"])
    requires_reference = grader_config.get("requires_reference", False)
    category = grader_config.get("category", "common")

    # Check if this is a multimodal grader (not supported for batch)
    if any(f in input_fields for f in MULTIMODAL_FIELDS):
        result.valid = False
        result.errors.append(
            "Multimodal graders are not supported for batch evaluation. "
            "Please use single evaluation mode for image-based graders."
        )
        return result

    # Determine required fields based on grader config
    required_fields = []

    if "response" in input_fields:
        required_fields.append("response")

    if "query" in input_fields:
        required_fields.append("query")

    if requires_reference or "reference_response" in input_fields:
        required_fields.append("reference_response")

    # Agent grader specific fields
    if "tool_definitions" in input_fields:
        required_fields.extend(["tool_definitions", "tool_calls"])
        if "reference_tool_calls" in input_fields:
            required_fields.append("reference_tool_calls")

    # Validate each record
    missing_fields_count = {}
    invalid_json_count = 0

    for i, record in enumerate(data):
        record_num = i + 1

        for field_name in required_fields:
            value = record.get(field_name)

            # Check if field is missing or empty
            if value is None or (isinstance(value, str) and not value.strip()):
                if field_name not in missing_fields_count:
                    missing_fields_count[field_name] = []
                if len(missing_fields_count[field_name]) < 5:  # Limit examples
                    missing_fields_count[field_name].append(record_num)

        # For agent graders, validate JSON fields
        if category == "agent":
            for json_field in ["tool_definitions", "tool_calls", "reference_tool_calls"]:
                if json_field in required_fields:
                    value = record.get(json_field)
                    if value and isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            # Update record with parsed value
                            record[json_field] = parsed
                        except json.JSONDecodeError:
                            invalid_json_count += 1
                            if invalid_json_count <= 3:
                                result.warnings.append(f"Record {record_num}: Invalid JSON in '{json_field}'")

    # Report missing required fields
    for field_name, record_nums in missing_fields_count.items():
        if len(record_nums) == len(data):
            result.valid = False
            result.errors.append(f"Required field '{field_name}' is missing in all records")
        elif record_nums:
            examples = ", ".join(str(n) for n in record_nums[:5])
            suffix = f" and {len(record_nums) - 5} more" if len(record_nums) > 5 else ""
            result.warnings.append(f"Field '{field_name}' is empty in records: {examples}{suffix}")

    if invalid_json_count > 3:
        result.warnings.append(f"... and {invalid_json_count - 3} more records with invalid JSON fields")

    return result


def get_required_fields_for_grader(grader_config: dict[str, Any]) -> list[str]:
    """Get list of required fields for a grader.

    Args:
        grader_config: Grader configuration from registry

    Returns:
        List of required field names
    """
    input_fields = grader_config.get("input_fields", ["query", "response"])
    requires_reference = grader_config.get("requires_reference", False)

    required = []

    if "response" in input_fields:
        required.append("response")

    if "query" in input_fields:
        required.append("query")

    if requires_reference:
        required.append("reference_response")

    if "tool_definitions" in input_fields:
        required.extend(["tool_definitions", "tool_calls"])

    if "reference_tool_calls" in input_fields:
        required.append("reference_tool_calls")

    return required


def get_optional_fields_for_grader(grader_config: dict[str, Any]) -> list[str]:
    """Get list of optional fields for a grader.

    Args:
        grader_config: Grader configuration from registry

    Returns:
        List of optional field names
    """
    input_fields = grader_config.get("input_fields", ["query", "response"])
    requires_reference = grader_config.get("requires_reference", False)

    optional = []

    # Context is always optional if supported
    if "context" in input_fields or "query" in input_fields:
        optional.append("context")

    # Reference is optional if not required
    if not requires_reference and "reference_response" in input_fields:
        optional.append("reference_response")

    return optional


def is_grader_batch_supported(grader_config: dict[str, Any]) -> tuple[bool, str]:
    """Check if a grader supports batch evaluation.

    Args:
        grader_config: Grader configuration from registry

    Returns:
        Tuple of (is_supported, reason_if_not)
    """
    input_fields = grader_config.get("input_fields", [])

    # Multimodal graders not supported
    if any(f in input_fields for f in MULTIMODAL_FIELDS):
        return False, "Multimodal graders require image input and are not supported for batch evaluation"

    return True, ""


def generate_sample_data(grader_config: dict[str, Any], format_type: str = "json") -> str:
    """Generate sample data file content for a grader.

    Args:
        grader_config: Grader configuration from registry
        format_type: 'json' or 'csv'

    Returns:
        Sample file content as string
    """
    required = get_required_fields_for_grader(grader_config)
    optional = get_optional_fields_for_grader(grader_config)
    category = grader_config.get("category", "common")

    # Create sample records
    if category == "agent":
        sample_records = [
            {
                "query": "What's the weather in Beijing?",
                "tool_definitions": json.dumps(
                    [
                        {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {"location": {"type": "string"}},
                        }
                    ]
                ),
                "tool_calls": json.dumps([{"name": "get_weather", "arguments": {"location": "Beijing"}}]),
            },
            {
                "query": "Calculate 15 * 23",
                "tool_definitions": json.dumps(
                    [
                        {
                            "name": "calculator",
                            "description": "Perform calculations",
                            "parameters": {"expression": {"type": "string"}},
                        }
                    ]
                ),
                "tool_calls": json.dumps([{"name": "calculator", "arguments": {"expression": "15 * 23"}}]),
            },
        ]
        if "reference_tool_calls" in required:
            sample_records[0]["reference_tool_calls"] = sample_records[0]["tool_calls"]
            sample_records[1]["reference_tool_calls"] = sample_records[1]["tool_calls"]
    else:
        sample_records = [
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
            },
            {
                "query": "Explain photosynthesis briefly.",
                "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
            },
        ]

        if "reference_response" in required:
            sample_records[0]["reference_response"] = "Paris is the capital of France."
            sample_records[1][
                "reference_response"
            ] = "Photosynthesis converts light energy to chemical energy in plants."

        if "context" in optional:
            sample_records[0]["context"] = "Geography question about European countries."

    if format_type == "json":
        return json.dumps({"data": sample_records}, indent=2, ensure_ascii=False)
    else:
        # CSV format
        if not sample_records:
            return ""

        output = io.StringIO()
        fieldnames = list(sample_records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for record in sample_records:
            writer.writerow(record)
        return output.getvalue()

"""Validate the small JSON Schema subset used by this protocol, without new venv dependencies.

Not a general JSON Schema implementation. Unsupported keywords are rejected so a future
schema revision cannot silently lose validation. Structural and numerical Cajal checks
remain in contracts.validate_bundle.
"""

import json
import math
import re

KEYWORDS = {
    "type",
    "properties",
    "additionalProperties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "enum",
    "const",
    "minimum",
    "maximum",
}


def validate(value, schema, path="$"):
    if set(schema) - KEYWORDS:
        raise ValueError("unsupported schema keyword")
    types = {
        "object": (dict,),
        "array": (list,),
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
    }
    if "type" in schema and type(value) not in types[schema["type"]]:
        raise ValueError(f"{path}: wrong JSON type")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite JSON number")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: wrong constant")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ValueError(f"{path}: invalid enum")
    if isinstance(value, dict):
        if not set(schema.get("required", [])) <= set(value):
            raise ValueError(f"{path}: missing required field")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: JSON object keys must be strings")
            if key in properties:
                validate(item, properties[key], path + "." + key)
            elif extra is False:
                raise ValueError(f"{path}: unknown field {key}")
            elif isinstance(extra, dict):
                validate(item, extra, path + "." + key)
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ValueError(f"{path}: invalid array length")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ValueError(f"{path}: duplicate array item")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf")):
            raise ValueError(f"{path}: invalid string length")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path}: pattern mismatch")
    if type(value) in (int, float) and not schema.get("minimum", -float("inf")) <= value <= schema.get(
        "maximum", float("inf")
    ):
        raise ValueError(f"{path}: number outside bounds")

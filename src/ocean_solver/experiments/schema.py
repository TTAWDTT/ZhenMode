"""YAML loading and small, explicit schema contracts; no numerical imports."""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    """An experiment cannot be resolved without changing its declared meaning."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject repeated YAML keys instead of silently choosing the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigurationError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def fields(value, required, optional=(), *, where="document"):
    if not isinstance(value, dict):
        raise ConfigurationError(f"{where} must be a mapping")
    unknown = set(value) - set(required) - set(optional)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ConfigurationError(f"{where}: unknown fields {sorted(unknown)}, missing {sorted(missing)}")
    return value


def identifier(value, where="id"):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value):
        raise ConfigurationError(f"{where} must be a stable lowercase identifier")
    return value


def text_value(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{where} must be a nonempty string")
    return value


def number(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{where} must be finite numeric")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ConfigurationError(f"{where} has an invalid sign")
    return value


def quantity(value, unit, where, *, positive=False, nonnegative=False, vector=False):
    fields(value, ("value", "unit"), where=where)
    if value["unit"] != unit:
        raise ConfigurationError(f"{where} requires unit {unit!r}; got {value['unit']!r}")
    values = value["value"] if vector else [value["value"]]
    if not isinstance(values, list) or not values:
        raise ConfigurationError(f"{where}.value must be a nonempty list")
    for item in values:
        number(item, where, positive=positive, nonnegative=nonnegative)
    return values if vector else values[0]


def load_document(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"missing configuration file: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: document must be a mapping")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ConfigurationError(f"{path}: schema_version must be 1")
    return value


def reference_path(root, reference):
    text_value(reference, "file reference")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()

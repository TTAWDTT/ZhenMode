"""Keep historical pickle names without importing any legacy facade."""


def preserve_legacy_names(namespace, legacy_name):
    """Retain names for definitions, while legacy imports alias this same module."""
    module_name = namespace["__name__"]
    for definition in namespace.values():
        if (getattr(definition, "__module__", None) == module_name
                and hasattr(definition, "__qualname__")):
            definition.__module__ = legacy_name

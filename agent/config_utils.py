import json
from pathlib import Path


try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is missing.
    yaml = None


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs.")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def dump_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"} and yaml is not None:
        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(config, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def section(config, name):
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return value


def get_nested(config, keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_path(value, base_dir=None):
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if base_dir is not None:
        return Path(base_dir) / path
    return path

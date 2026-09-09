# Recipe loading shared by the entry points (train, eval): one YAML
# dialect and one override syntax, so a recipe reads the same whichever
# stage consumes it and a launch-time `key=value` means the same thing
# everywhere.

import re
from pathlib import Path

import yaml

# YAML 1.1 parses `1e-4` (no dot) as a *string*; with args passed through
# as plain dicts there is no dataclass layer to coerce it back. Register
# the full float form as an implicit resolver once, globally.
_FLOAT_RE = re.compile(r'''^[-+]?(
    (\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? | \d+[eE][-+]?\d+
    )$''', re.X)


class YamlLoader(yaml.SafeLoader):
    pass


YamlLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float', _FLOAT_RE, list('-+0123456789.'))


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.load(f, YamlLoader) or {}


def parse_value(text: str):
    '''The value side of an override, parsed as yaml (`4`, `float32`,
    `[a, b]`, `null`).'''
    return yaml.load(text, YamlLoader)


def apply_overrides(raw: dict, overrides: list[str] | tuple[str, ...]) -> dict:
    '''`key=value` strings applied to a loaded recipe dict, for launch-time
    variations of one recipe (smoke runs, a kernel A/B): the recipe file
    stays the record, the resolved config written to the run dir carries
    the result. A dotted key reaches one level into a dict field
    (`model_args.la_fused=true`). Returns a new dict.'''
    raw = dict(raw)
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise ValueError(f'override {item!r} is not key=value')
        outer, dot, inner = key.partition('.')
        if not dot:
            raw[key] = parse_value(value)
            continue
        if not isinstance(raw.get(outer), dict):
            raise ValueError(f'override {item!r}: {outer} is not a dict field')
        raw[outer] = dict(raw[outer]) | {inner: parse_value(value)}
    return raw

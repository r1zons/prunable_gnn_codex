"""Very small YAML fallback parser/dumper for simple mapping files.

This module supports only the subset of YAML used by project config files:
- nested dictionaries using spaces for indentation
- scalar values (str/int/float/bool)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def safe_load(text: str) -> Dict[str, Any]:
    """Parse minimal YAML text into a dictionary."""
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]

    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Invalid indentation: {line}")

        key, raw_value = _parse_line(line.strip())
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if raw_value is None:
            child: Dict[str, Any] = {}
            current[key] = child
            stack.append((indent + 2, child))
        else:
            current[key] = _parse_scalar(raw_value)

    return root


def safe_dump(payload: Dict[str, Any], sort_keys: bool = False) -> str:
    """Serialize dictionary into minimal YAML text."""
    lines: List[str] = []
    _dump_map(payload, lines, indent=0, sort_keys=sort_keys)
    return "\n".join(lines) + "\n"


def _dump_map(payload: Dict[str, Any], lines: List[str], indent: int, sort_keys: bool) -> None:
    keys = sorted(payload) if sort_keys else payload.keys()
    for key in keys:
        value = payload[key]
        prefix = " " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            _dump_map(value, lines, indent=indent + 2, sort_keys=sort_keys)
        else:
            lines.append(f"{prefix}{key}: {_format_scalar(value)}")


def _parse_line(line: str) -> Tuple[str, Optional[str]]:
    if ":" not in line:
        raise ValueError(f"Invalid YAML line: {line}")
    key, rest = line.split(":", 1)
    key = key.strip()
    value = rest.strip()
    return key, value if value != "" else None


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

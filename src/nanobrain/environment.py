"""Read local dataset settings without executing shell code."""

import re
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)


def _decode_dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if raw.strip().startswith('"'):
            value = value.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")
            value = value.replace(r'\"', '"').replace(r"\\", "\\")
        else:
            value = value.replace("\\'", "'")
        return value
    # Whitespace separates inline comments; URL fragments remain intact.
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def read_env_file(path: str | Path) -> dict[str, str]:
    """Read dotenv settings, ignoring missing files and malformed lines."""
    target = Path(path).expanduser()
    if not target.is_file():
        return {}
    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        key, raw = match.groups()
        if _KEY_RE.fullmatch(key) is None:
            continue
        values[key] = _decode_dotenv_value(raw)
    return values

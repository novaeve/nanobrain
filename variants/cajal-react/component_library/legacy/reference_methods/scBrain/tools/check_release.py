#!/usr/bin/env python3
"""Validate that the scBrain code release is source-only and syntactically sound."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PATTERNS = [
    re.compile("/data" + "1/"),
    re.compile("/data" + "2/"),
    re.compile("/ro" + "ot/"),
    re.compile("/ho" + "me/"),
    re.compile("/tm" + "p/"),
]
FORBIDDEN_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".npz",
    ".npy",
    ".pkl",
    ".h5ad",
    ".tar.gz",
}
FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "checkpoints_origin",
}


def is_forbidden_artifact(path: Path) -> bool:
    name = path.name
    return any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def check_required_dirs(errors: list[str]) -> None:
    for rel in ["src", "experiments", "paper_tools", "configs", "data", "weights", "outputs", "docs"]:
        if not (ROOT / rel).exists():
            errors.append(f"Missing required directory: {rel}")


def check_no_large_or_binary_artifacts(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if is_forbidden_artifact(path):
            errors.append(f"Forbidden data/weight artifact in release: {rel}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"File exceeds 5MB release limit: {rel}")


def check_no_generated_dirs(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            errors.append(f"Generated directory should not be in release tree: {path.relative_to(ROOT)}")


def check_no_local_absolute_paths(errors: list[str]) -> None:
    text_suffixes = {".py", ".sh", ".md", ".txt", ".json", ".yml", ".yaml", ".log", ".csv", ".tex"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        text = path.read_text(errors="ignore")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"Local absolute path remains in {path.relative_to(ROOT)}: {pattern.pattern}")


def check_python_syntax(errors: list[str]) -> None:
    for path in ROOT.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc.msg}")


def main() -> int:
    errors: list[str] = []
    check_required_dirs(errors)
    check_no_large_or_binary_artifacts(errors)
    check_no_generated_dirs(errors)
    check_no_local_absolute_paths(errors)
    check_python_syntax(errors)

    if errors:
        print("Release check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Release check passed: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

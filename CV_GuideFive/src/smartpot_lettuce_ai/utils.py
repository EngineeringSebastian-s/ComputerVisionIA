"""Shared utility helpers for the project."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: Path | str) -> Any:
    """Load JSON data from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path | str, *, indent: int = 2) -> None:
    """Persist JSON data to disk using UTF-8 encoding."""

    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def slugify_label(value: str) -> str:
    """Convert a label into a filesystem-friendly slug."""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug or "item"


def folderify_label(value: str) -> str:
    """Convert a human-readable class name into the folder name used by the dataset."""

    folder_name = re.sub(r"[\s-]+", "_", value.strip())
    folder_name = re.sub(r"_+", "_", folder_name)
    return folder_name


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the repository root by walking up until `pyproject.toml` is found."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    return current


def set_global_seed(seed: int) -> None:
    """Seed the common random generators used by the project."""

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a project dependency
        np = None
    else:
        np.random.seed(seed)

    try:
        import tensorflow as tf
    except Exception:  # pragma: no cover - optional until training time
        return

    tf.random.set_seed(seed)

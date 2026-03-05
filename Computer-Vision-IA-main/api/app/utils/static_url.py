from __future__ import annotations

from pathlib import Path


def build_static_url(path: Path) -> str:
    output_root = Path("app/output")
    if path.is_absolute():
        relative_path = path.resolve().relative_to(output_root.resolve())
    else:
        relative_path = path.relative_to(output_root)
    return f"/static/{relative_path.as_posix()}"

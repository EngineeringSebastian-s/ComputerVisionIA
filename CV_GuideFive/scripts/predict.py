from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    _bootstrap_src_path()

    from smartpot_lettuce_ai.cli import main as cli_main

    return cli_main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

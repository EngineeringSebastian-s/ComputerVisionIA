"""SmartPot lettuce disease classification toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("final-vision-computadora")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/local runs
    __version__ = "0.1.0"

__all__ = ["__version__"]

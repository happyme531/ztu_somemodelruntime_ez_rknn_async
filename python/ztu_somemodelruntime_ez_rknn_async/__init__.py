"""Python bindings for ztu_somemodelruntime_ez_rknn_async."""

from pathlib import Path
import re

from ._core import InferenceSession, ModelMetadata, NodeArg
from .options import RknnProviderOptions, make_provider_options

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover - Python 3.7 fallback
    PackageNotFoundError = Exception
    _pkg_version = None

__all__ = [
    "InferenceSession",
    "ModelMetadata",
    "NodeArg",
    "RknnProviderOptions",
    "make_provider_options",
]


def _read_version_from_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0+unknown"
    match = re.search(r'(?m)^version = "([^"]+)"$', text)
    if match is None:
        return "0+unknown"
    return match.group(1)


try:
    if _pkg_version is None:
        raise PackageNotFoundError
    __version__ = _pkg_version("ztu_somemodelruntime_ez_rknn_async")
except PackageNotFoundError:
    __version__ = _read_version_from_pyproject()

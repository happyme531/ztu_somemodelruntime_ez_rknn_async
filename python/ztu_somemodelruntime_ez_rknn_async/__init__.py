"""Python bindings for ztu_somemodelruntime_ez_rknn_async."""

from ._core import InferenceSession, NodeArg

# TODO: expose custom op loading APIs when needed.

__all__ = ["InferenceSession", "NodeArg"]
__version__ = "0.2.0"

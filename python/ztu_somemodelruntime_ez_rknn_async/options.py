from __future__ import annotations

from os import PathLike
from typing import Sequence, TypedDict, Union

try:
    from typing import Literal
except ImportError:  # pragma: no cover - Python 3.7 fallback
    from typing_extensions import Literal

PathLikeStr = Union[str, PathLike[str]]
ScheduleLike = Union[int, str, Sequence[int]]
TpModeLike = Literal["auto", "all", "0", "1", "2", "0,1", "0,1,2"]
LayoutLike = Literal[
    "nchw",
    "original",
    "nchw_software",
    "original_software",
    "nhwc",
    "any",
]


class RknnProviderOptions(TypedDict, total=False):
    """Provider options for RKNN runtime.

    Notes:
    - Unknown keys are rejected at runtime.
    - ``custom_op_path`` and ``custom_op_paths`` are aliases.
    - ``custom_op_default_path`` and ``load_custom_ops_from_default_path``
      are aliases.
    """

    layout: LayoutLike
    max_queue_size: int
    threads_per_core: int
    sequential_callbacks: bool
    schedule: ScheduleLike
    tp_mode: TpModeLike
    enable_pacing: bool
    disable_dup_context: bool
    custom_op_path: Union[PathLikeStr, Sequence[PathLikeStr]]
    custom_op_paths: Union[PathLikeStr, Sequence[PathLikeStr]]
    custom_op_default_path: bool
    load_custom_ops_from_default_path: bool


def make_provider_options(
    *,
    layout: LayoutLike = "nchw_software",
    max_queue_size: int = 3,
    threads_per_core: int = 1,
    sequential_callbacks: bool = True,
    schedule: Union[ScheduleLike, None] = None,
    tp_mode: Union[TpModeLike, None] = None,
    enable_pacing: bool = False,
    disable_dup_context: bool = False,
    custom_op_paths: Union[PathLikeStr, Sequence[PathLikeStr], None] = None,
    custom_op_default_path: bool = False,
) -> RknnProviderOptions:
    """Build typed RKNN provider options for ``InferenceSession``.

    Args:
        layout: Input layout policy. Common values are ``"nhwc"`` and
            ``"nchw_software"``.
        max_queue_size: Max pending task count in async queue, must be ``> 0``.
        threads_per_core: Worker thread count per selected NPU core, must be
            ``> 0``.
        sequential_callbacks: If True, async callbacks are emitted in submit
            order.
        schedule: Core schedule (data-parallel mode). Accepts int, string
            (e.g. ``"0,1,2"``), or sequence of ints.
        tp_mode: Tensor-parallel core-mask mode. Supported values are
            ``"auto"``, ``"all"``, ``"0"``, ``"1"``, ``"2"``,
            ``"0,1"``, and ``"0,1,2"``.
        enable_pacing: Enable input pacing based on recent throughput.
        disable_dup_context: If True, avoid ``rknn_dup_context`` and initialize
            each worker context independently.
        custom_op_paths: One path or multiple ``.so`` plugin paths.
        custom_op_default_path: If True, scan RKNN default plugin path.

    Returns:
        A ``dict`` suitable for ``provider_options``.

    Notes:
        ``schedule`` and ``tp_mode`` are mutually exclusive.
        Requesting custom-op loading will force disable-dup-context behavior in
        runtime even if ``disable_dup_context`` is False.
    """
    if schedule is not None and tp_mode is not None:
        raise ValueError("provider options 'tp_mode' conflicts with 'schedule'; set only one.")

    options: RknnProviderOptions = {
        "layout": layout,
        "max_queue_size": max_queue_size,
        "threads_per_core": threads_per_core,
        "sequential_callbacks": sequential_callbacks,
        "enable_pacing": enable_pacing,
        "disable_dup_context": disable_dup_context,
        "custom_op_default_path": custom_op_default_path,
    }
    if schedule is not None:
        options["schedule"] = schedule
    if tp_mode is not None:
        options["tp_mode"] = tp_mode
    if custom_op_paths is not None:
        options["custom_op_paths"] = custom_op_paths
    return options

from __future__ import annotations

from os import PathLike
from typing import Literal, Sequence, TypedDict, Union

PathLikeStr = Union[str, PathLike[str]]
ScheduleLike = Union[int, str, Sequence[int]]
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
    schedule: ScheduleLike = (0,),
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
        schedule: Core schedule. Accepts int, string (e.g. ``"0,1,2"``), or
            sequence of ints.
        enable_pacing: Enable input pacing based on recent throughput.
        disable_dup_context: If True, avoid ``rknn_dup_context`` and initialize
            each worker context independently.
        custom_op_paths: One path or multiple ``.so`` plugin paths.
        custom_op_default_path: If True, scan RKNN default plugin path.

    Returns:
        A ``dict`` suitable for ``provider_options``.

    Notes:
        Requesting custom-op loading will force disable-dup-context behavior in
        runtime even if ``disable_dup_context`` is False.
    """
    options: RknnProviderOptions = {
        "layout": layout,
        "max_queue_size": max_queue_size,
        "threads_per_core": threads_per_core,
        "sequential_callbacks": sequential_callbacks,
        "schedule": schedule,
        "enable_pacing": enable_pacing,
        "disable_dup_context": disable_dup_context,
        "custom_op_default_path": custom_op_default_path,
    }
    if custom_op_paths is not None:
        options["custom_op_paths"] = custom_op_paths
    return options

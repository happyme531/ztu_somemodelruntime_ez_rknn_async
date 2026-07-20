from __future__ import annotations

from os import PathLike
from typing import Sequence, Union

try:
    from typing import Literal, TypedDict
except ImportError:  # pragma: no cover - Python 3.7 fallback
    from typing_extensions import Literal, TypedDict

PathLikeStr = Union[str, PathLike]
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
ZeroCopyLayoutLike = Literal["nhwc", "native"]


class RknnProviderOptions(TypedDict, total=False):
    """Provider options for the RKNPU2 runtime (legacy name).

    Notes:
    - Unknown keys are rejected at runtime.
    - ``custom_op_path`` and ``custom_op_paths`` are aliases.
    - ``custom_op_default_path`` and ``load_custom_ops_from_default_path``
      are aliases.
    """

    layout: LayoutLike
    zero_copy_layout: ZeroCopyLayoutLike
    max_queue_size: int
    threads_per_core: int
    submit_timeout_ms: int
    sequential_callbacks: bool
    schedule: ScheduleLike
    tp_mode: TpModeLike
    enable_pacing: bool
    disable_dup_context: bool
    custom_op_path: Union[PathLikeStr, Sequence[PathLikeStr]]
    custom_op_paths: Union[PathLikeStr, Sequence[PathLikeStr]]
    custom_op_default_path: bool
    load_custom_ops_from_default_path: bool


Rknpu2ProviderOptions = RknnProviderOptions


class Rknpu3ProviderOptions(TypedDict, total=False):
    """Provider options specific to ``RKNPU3ExecutionProvider``."""

    runtime_library: PathLikeStr
    device_id: Union[str, int]
    weight_path: PathLikeStr
    priority: int
    run_timeout_ms: int
    run_core_mask: Union[int, Literal["auto", "all"]]


def make_provider_options(
    *,
    layout: LayoutLike = "original",
    zero_copy_layout: ZeroCopyLayoutLike = "nhwc",
    max_queue_size: int = 3,
    threads_per_core: int = 1,
    submit_timeout_ms: int = 10000,
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
        layout: Input layout policy. Common values are ``"nchw"(aka original)`` and
            ``"nhwc"``.
        zero_copy_layout: Native I/O binding layout policy, ``"nhwc"`` or
            ``"native"``.
        max_queue_size: Max pending task count in async queue, must be ``> 0``.
        threads_per_core: Worker thread count per selected NPU core, must be
            ``> 0``.
        submit_timeout_ms: Max wait time (milliseconds) when submit is blocked
            by full task/callback queues.
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
        "zero_copy_layout": zero_copy_layout,
        "max_queue_size": max_queue_size,
        "threads_per_core": threads_per_core,
        "submit_timeout_ms": submit_timeout_ms,
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


make_rknpu2_provider_options = make_provider_options


def make_rknpu3_provider_options(
    *,
    runtime_library: Union[PathLikeStr, None] = None,
    device_id: Union[str, int, None] = None,
    weight_path: Union[PathLikeStr, None] = None,
    priority: int = 0,
    run_timeout_ms: int = 0,
    run_core_mask: Union[int, Literal["auto", "all"]] = "auto",
) -> Rknpu3ProviderOptions:
    """Build options for the synchronous, native zero-copy RKNPU3 provider.

    ``weight_path`` normally need not be supplied for a path-based model: the
    provider requires and automatically selects the same basename with the
    ``.weight`` suffix. It is required when ``InferenceSession`` receives model
    bytes. Refusing to run without weights is intentional because RKNN3's
    streaming mode can otherwise return success with incorrect outputs.

    ``run_core_mask="auto"`` queries the model core count and builds a
    contiguous mask. This works around RKCP runtime 1.0.4 rejecting the
    documented numeric AUTO value (zero).
    """

    options: Rknpu3ProviderOptions = {
        "priority": priority,
        "run_timeout_ms": run_timeout_ms,
        "run_core_mask": run_core_mask,
    }
    if runtime_library is not None:
        options["runtime_library"] = runtime_library
    if device_id is not None:
        options["device_id"] = device_id
    if weight_path is not None:
        options["weight_path"] = weight_path
    return options

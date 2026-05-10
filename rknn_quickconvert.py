#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _build_config_kwargs(
    target_platform: str,
    remove_weight: bool,
    config_kwargs: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "quantized_algorithm": "normal",
        "quantized_method": "channel",
        "target_platform": target_platform,
        "optimization_level": 3,
        "model_pruning": True,
        "remove_weight": remove_weight,
    }
    if config_kwargs is not None:
        kwargs.update(dict(config_kwargs))
    kwargs["target_platform"] = target_platform
    kwargs["remove_weight"] = remove_weight
    return kwargs


def quick_convert(
    onnx_model_path: str,
    *,
    rknn_model_path: Optional[str] = None,
    remove_weight: bool = False,
    target_platform: str = "rk3588",
    verbose: bool = True,
    config_kwargs: Optional[Mapping[str, Any]] = None,
    load_kwargs: Optional[Mapping[str, Any]] = None,
    build_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    from rknn.api import RKNN

    onnx_path = Path(onnx_model_path)
    if rknn_model_path is None:
        rknn_model_path = str(onnx_path.with_suffix(".rknn"))

    rknn = RKNN(verbose=verbose)
    try:
        print("--> Config model")
        ret = rknn.config(
            **_build_config_kwargs(target_platform, remove_weight, config_kwargs)
        )
        if ret != 0:
            raise RuntimeError("Config model failed!")

        print("--> Loading model")
        load_args: Dict[str, Any] = {"model": str(onnx_path)}
        if load_kwargs is not None:
            load_args.update(dict(load_kwargs))
        ret = rknn.load_onnx(**load_args)
        if ret != 0:
            raise RuntimeError("Load model failed!")

        print("--> Building model")
        build_args: Dict[str, Any] = {
            "do_quantization": False,
            "dataset": None,
            "rknn_batch_size": None,
        }
        if build_kwargs is not None:
            build_args.update(dict(build_kwargs))
        ret = rknn.build(**build_args)
        if ret != 0:
            raise RuntimeError("Build model failed!")

        print("--> Export RKNN model")
        ret = rknn.export_rknn(rknn_model_path)
        if ret != 0:
            raise RuntimeError("Export RKNN model failed!")

        print(f"RKNN model exported to {rknn_model_path}")
        return rknn_model_path
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model_path", type=str, help="Path to the ONNX model")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output RKNN path (defaults to <input>.rknn)",
    )
    parser.add_argument(
        "--remove-weight",
        action="store_true",
        help="Strip weights during conversion for shared-weight deployment",
    )
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        help="Target RKNN platform",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce RKNN verbosity",
    )
    args = parser.parse_args()

    quick_convert(
        args.onnx_model_path,
        rknn_model_path=args.output,
        remove_weight=args.remove_weight,
        target_platform=args.target_platform,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

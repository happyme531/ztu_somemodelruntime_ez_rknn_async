import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("rknn.api")
ezrknn = pytest.importorskip("ztu_somemodelruntime_ez_rknn_async")
from onnx import TensorProto, helper


RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NC1HWC2 = 2


def _load_quick_convert():
    project_root = Path(__file__).resolve().parents[1]
    quickconvert_path = project_root.parent / "rknn_quickconvert.py"
    spec = importlib.util.spec_from_file_location("rknn_quickconvert", quickconvert_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load {quickconvert_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.quick_convert


@pytest.fixture(scope="session")
def _layout_mode_model(tmp_path_factory):
    shape = (1, 8, 64, 64)
    root = tmp_path_factory.mktemp("zero_copy_layout_mode")
    onnx_path = root / "add.onnx"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT16, shape)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"])],
        "add_zero_copy_layout_mode",
        [x, y],
        [z],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, onnx_path)

    quick_convert = _load_quick_convert()
    old_cwd = os.getcwd()
    try:
        os.chdir(root)
        quick_convert(str(onnx_path))
    except SystemExit as exc:
        pytest.skip(f"RKNN conversion failed for zero_copy_layout test: {exc}")
    finally:
        os.chdir(old_cwd)

    rknn_path = onnx_path.with_suffix(".rknn")
    if not rknn_path.exists():
        pytest.skip(f"RKNN conversion did not produce {rknn_path}")
    return str(rknn_path), shape


def test_zero_copy_layout_mode_selects_nhwc_or_native_attrs(_layout_mode_model):
    model_path, shape = _layout_mode_model

    default_session = ezrknn.InferenceSession(model_path)
    default_binding = default_session.io_binding()
    default_binding.bind_output("z", "rknpu2")
    assert default_binding.get_outputs()[0].memory_info()["fmt"] == RKNN_TENSOR_NHWC

    native_session = ezrknn.InferenceSession(
        model_path, provider_options={"zero_copy_layout": "native"}
    )
    native_binding = native_session.io_binding()
    native_binding.bind_output("z", "rknpu2")
    native_output = native_binding.get_outputs()[0]
    assert native_output.memory_info()["fmt"] == RKNN_TENSOR_NC1HWC2

    native_input = ezrknn.OrtValue.ortvalue_from_shape_and_type(
        shape,
        np.float16,
        "rknpu2",
        session=native_session,
        name="x",
        io_kind="input",
    )
    assert native_input.memory_info()["fmt"] == RKNN_TENSOR_NC1HWC2
    assert native_input.memory_info()["pass_through"]

    native_binding.bind_ortvalue_input("x", native_output)

    generic_input = ezrknn.OrtValue.ortvalue_from_shape_and_type(
        shape, np.float16, "rknpu2"
    )
    with pytest.raises(RuntimeError, match="native zero-copy layout mismatch"):
        native_binding.bind_ortvalue_input("x", generic_input)


def test_zero_copy_layout_mode_rejects_unknown_option(_layout_mode_model):
    model_path, _ = _layout_mode_model
    with pytest.raises(RuntimeError, match="Unknown zero_copy_layout"):
        ezrknn.InferenceSession(
            model_path, provider_options={"zero_copy_layout": "blocked"}
        )

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("rknn.api")
ezrknn = pytest.importorskip("ztu_somemodelruntime_ez_rknn_async")
from onnx import TensorProto, helper


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
def quick_convert():
    return _load_quick_convert()


def _write_add_onnx(path, shape):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT16, shape)
    node = helper.make_node("Add", ["x", "y"], ["z"])
    graph = helper.make_graph([node], "add_copy_layout", [x, y], [z])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


@pytest.fixture(scope="session")
def nchw_add_model(tmp_path_factory, quick_convert):
    case_dir = tmp_path_factory.mktemp("copy_layout_add")
    onnx_path = case_dir / "add.onnx"
    _write_add_onnx(onnx_path, [1, 3, 64, 64])

    old_cwd = os.getcwd()
    try:
        os.chdir(case_dir)
        quick_convert(str(onnx_path))
    except SystemExit as exc:
        pytest.skip(f"RKNN conversion failed for copy layout test: {exc}")
    finally:
        os.chdir(old_cwd)

    rknn_path = onnx_path.with_suffix(".rknn")
    if not rknn_path.exists():
        pytest.skip(f"RKNN conversion did not produce {rknn_path}")
    return str(rknn_path), (1, 3, 64, 64)


def _make_inputs(shape):
    numel = int(np.prod(shape))
    base = np.arange(numel, dtype=np.int64)
    x = ((base % 23) - 11).reshape(shape).astype(np.float16)
    y = (((base * 3) % 29) - 14).reshape(shape).astype(np.float16)
    return x, y


@pytest.mark.parametrize(
    "layout,input_layout",
    [
        ("nchw", "nchw"),
        ("nchw_software", "nchw"),
        ("nhwc", "nhwc"),
    ],
)
def test_copy_run_layout_inputs(nchw_add_model, layout, input_layout):
    model_path, model_shape = nchw_add_model
    session = ezrknn.InferenceSession(model_path, provider_options={"layout": layout})

    nhwc_shape = (model_shape[0], model_shape[2], model_shape[3], model_shape[1])
    expected_input_shape = model_shape if input_layout == "nchw" else nhwc_shape
    assert tuple(session.get_inputs()[0].shape) == expected_input_shape

    x_nchw, y_nchw = _make_inputs(model_shape)
    if input_layout == "nhwc":
        x = np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)))
        y = np.ascontiguousarray(np.transpose(y_nchw, (0, 2, 3, 1)))
    else:
        x = x_nchw
        y = y_nchw

    actual = session.run(None, {"x": x, "y": y})[0].astype(np.float16, copy=False)
    expected = (x_nchw + y_nchw).astype(np.float16, copy=False)

    assert actual.shape == model_shape
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

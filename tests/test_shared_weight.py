import gc
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
    quickconvert_path = project_root / "rknn_quickconvert.py"
    spec = importlib.util.spec_from_file_location("rknn_quickconvert", quickconvert_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load {quickconvert_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.quick_convert


@pytest.fixture(scope="session")
def quick_convert():
    return _load_quick_convert()


def _write_mul_onnx(path, shape, scale):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT, scale.shape, scale.reshape(-1).tolist()
    )
    node = helper.make_node("Mul", ["x", "scale"], ["y"])
    graph = helper.make_graph([node], "shared_weight_mul", [x], [y], [scale_init])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


def _make_inputs(shape):
    base = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (base % 17).astype(np.float32) / 8.0 - 1.0


def _reference_mul(x, scale):
    return x * scale


def _run_iobinding_or_skip(session, binding):
    try:
        session.run_with_iobinding(binding)
    except RuntimeError as exc:
        message = str(exc)
        if "rknn_set_io_mem failed" in message:
            pytest.skip(f"RKNN runtime rejected this zero-copy binding: {message}")
        raise


@pytest.fixture(scope="session")
def shared_weight_models(tmp_path_factory, quick_convert):
    root = tmp_path_factory.mktemp("shared_weight")
    scale = np.array([[[1.0]], [[0.5]], [[-0.25]]], dtype=np.float32)

    master_shape = (3, 4, 16)
    slave_shape = (3, 4, 20)

    master_dir = root / "master"
    slave_dir = root / "slave"
    master_dir.mkdir()
    slave_dir.mkdir()

    master_onnx = master_dir / "shared_weight_master.onnx"
    slave_onnx = slave_dir / "shared_weight_slave.onnx"
    _write_mul_onnx(master_onnx, master_shape, scale)
    _write_mul_onnx(slave_onnx, slave_shape, scale)

    old_cwd = os.getcwd()
    try:
        os.chdir(master_dir)
        master_rknn = quick_convert(
            str(master_onnx),
            remove_weight=False,
            load_kwargs={"inputs": ["x"], "input_size_list": [list(master_shape)]},
        )
        os.chdir(slave_dir)
        slave_rknn = quick_convert(
            str(slave_onnx),
            remove_weight=True,
            load_kwargs={"inputs": ["x"], "input_size_list": [list(slave_shape)]},
        )
    except SystemExit as exc:
        pytest.skip(f"RKNN conversion failed for shared weight test: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"RKNN conversion failed for shared weight test: {exc}")
    finally:
        os.chdir(old_cwd)

    if not Path(master_rknn).exists() or not Path(slave_rknn).exists():
        pytest.skip("RKNN conversion did not produce shared-weight outputs")

    return {
        "scale": scale,
        "master_shape": master_shape,
        "slave_shape": slave_shape,
        "master_rknn": master_rknn,
        "slave_rknn": slave_rknn,
    }


def test_shared_weight_remove_weight_conversion_and_runtime(shared_weight_models):
    assert shared_weight_models["master_shape"] != shared_weight_models["slave_shape"]
    assert shared_weight_models["scale"].shape == (3, 1, 1)

    master_size = Path(shared_weight_models["master_rknn"]).stat().st_size
    slave_size = Path(shared_weight_models["slave_rknn"]).stat().st_size
    assert master_size > 0
    assert 0 < slave_size < master_size

    try:
        master_session = ezrknn.InferenceSession(shared_weight_models["master_rknn"])
        slave_session = ezrknn.InferenceSession(
            shared_weight_models["slave_rknn"], use_weight_from=master_session
        )
    except RuntimeError as exc:
        pytest.skip(f"RKNN runtime cannot load shared-weight models: {exc}")

    master_input = _make_inputs(shared_weight_models["master_shape"])
    slave_input = _make_inputs(shared_weight_models["slave_shape"])
    scale = shared_weight_models["scale"]

    master_expected = _reference_mul(master_input, scale)
    slave_expected = _reference_mul(slave_input, scale)

    master_output = master_session.run(None, {"x": master_input})[0]
    slave_output = slave_session.run(None, {"x": slave_input})[0]

    np.testing.assert_allclose(master_output, master_expected, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(slave_output, slave_expected, rtol=1e-2, atol=1e-2)

    del master_session
    gc.collect()

    slave_input_2 = _make_inputs(shared_weight_models["slave_shape"]) + 0.25
    slave_expected_2 = _reference_mul(slave_input_2, scale)
    slave_output_2 = slave_session.run(None, {"x": slave_input_2})[0]
    np.testing.assert_allclose(slave_output_2, slave_expected_2, rtol=1e-2, atol=1e-2)


def test_shared_weight_slave_supports_zero_copy_iobinding(shared_weight_models):
    try:
        master_session = ezrknn.InferenceSession(shared_weight_models["master_rknn"])
        slave_session = ezrknn.InferenceSession(
            shared_weight_models["slave_rknn"], use_weight_from=master_session
        )
    except RuntimeError as exc:
        pytest.skip(f"RKNN runtime cannot load shared-weight models: {exc}")

    slave_input = _make_inputs(shared_weight_models["slave_shape"]).astype(np.float16)
    scale = shared_weight_models["scale"]
    expected = _reference_mul(slave_input, scale)
    copy_baseline = slave_session.run(None, {"x": slave_input})[0]

    binding = slave_session.io_binding()
    x_value = ezrknn.OrtValue.ortvalue_from_numpy(
        slave_input, "rknpu2", session=slave_session, name="x", io_kind="input"
    )
    binding.bind_ortvalue_input("x", x_value)
    binding.bind_output("y", "rknpu2")
    y_value = binding.get_outputs()[0]

    _run_iobinding_or_skip(slave_session, binding)
    zero_copy_output = y_value.numpy()

    assert zero_copy_output.shape == slave_input.shape
    assert y_value.memory_info()["device_type"] == "rknpu2"
    assert y_value.memory_info()["dmabuf_backed"]
    np.testing.assert_allclose(zero_copy_output, copy_baseline, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(zero_copy_output, expected, rtol=1e-2, atol=1e-2)

    slave_input_2 = slave_input + np.asarray(0.25, dtype=slave_input.dtype)
    x_value.update_inplace(slave_input_2)
    _run_iobinding_or_skip(slave_session, binding)
    zero_copy_output_2 = y_value.numpy()
    np.testing.assert_allclose(
        zero_copy_output_2,
        _reference_mul(slave_input_2, scale),
        rtol=1e-2,
        atol=1e-2,
    )

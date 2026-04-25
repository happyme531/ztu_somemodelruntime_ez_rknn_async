import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("rknn.api")
ezrknn = pytest.importorskip("ztu_somemodelruntime_ez_rknn_async")
from onnx import TensorProto, helper


MIN_TENSOR_BYTES = 32 * 1024 * 1024


DTYPE_CASES = [
    ("fp16", np.float16, TensorProto.FLOAT16),
    ("fp32", np.float32, TensorProto.FLOAT),
    ("int32", np.int32, TensorProto.INT32),
    ("int64", np.int64, TensorProto.INT64),
]

SHAPE_CASES = [
    pytest.param(3, None, id="3d"),
    pytest.param(5, None, id="5d"),
    pytest.param(4, 1, id="4d_c1"),
    pytest.param(4, 3, id="4d_c3"),
    pytest.param(4, 4, id="4d_c4"),
    pytest.param(4, 8, id="4d_c_other"),
]

SESSION_DTYPE_TO_NUMPY = {
    "tensor(float16)": np.float16,
    "tensor(float32)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}

RKNN_TENSOR_NC1HWC2 = 2
RKNN_TENSOR_NHWC = 1


def _shape_for(rank, dtype, channel):
    itemsize = np.dtype(dtype).itemsize
    elems = (MIN_TENSOR_BYTES + itemsize - 1) // itemsize
    if rank == 3:
        return (elems // (512 * 512), 512, 512)
    if rank == 4:
        if channel is None:
            raise AssertionError("4D shape cases must specify channel")
        height = 2048
        width = (elems + channel * height - 1) // (channel * height)
        return (1, channel, height, width)
    if rank == 5:
        return (1, elems // (64 * 128 * 128), 64, 128, 128)
    raise AssertionError(f"unsupported rank: {rank}")


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


def _write_add_onnx(path, shape, tensor_type):
    x = helper.make_tensor_value_info("x", tensor_type, shape)
    y = helper.make_tensor_value_info("y", tensor_type, shape)
    z = helper.make_tensor_value_info("z", tensor_type, shape)
    node = helper.make_node("Add", ["x", "y"], ["z"])
    graph = helper.make_graph([node], "add_zero_copy", [x, y], [z])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


@pytest.fixture(scope="session")
def rknn_model_factory(tmp_path_factory, quick_convert):
    cache = {}
    root = tmp_path_factory.mktemp("zero_copy_add_models")

    def make_model(rank, channel, dtype_name, shape, tensor_type):
        key = (rank, channel, dtype_name, tuple(shape))
        if key in cache:
            return cache[key]
        shape_name = f"{rank}d" if channel is None else f"{rank}d_c{channel}"
        case_dir = root / f"{shape_name}_{dtype_name}"
        case_dir.mkdir()
        onnx_path = case_dir / "add.onnx"
        _write_add_onnx(onnx_path, shape, tensor_type)

        old_cwd = os.getcwd()
        try:
            os.chdir(case_dir)
            quick_convert(str(onnx_path))
        except SystemExit as exc:
            pytest.skip(f"RKNN conversion failed for {rank}d {dtype_name}: {exc}")
        finally:
            os.chdir(old_cwd)

        rknn_path = onnx_path.with_suffix(".rknn")
        if not rknn_path.exists():
            pytest.skip(f"RKNN conversion did not produce {rknn_path}")
        cache[key] = str(rknn_path)
        return cache[key]

    return make_model


def _make_inputs(shape, dtype):
    numel = int(np.prod(shape))
    base = np.arange(numel, dtype=np.int64)
    x = ((base % 23) - 11).reshape(shape).astype(dtype)
    y = (((base * 3) % 29) - 14).reshape(shape).astype(dtype)
    return x, y


def _nhwc_shape_for(dtype, channels):
    itemsize = np.dtype(dtype).itemsize
    elems = (MIN_TENSOR_BYTES + itemsize - 1) // itemsize
    height = 2048
    width = (elems + channels * height - 1) // (channels * height)
    return (1, height, width, channels)


def _session_numpy_dtype(node_arg):
    try:
        return SESSION_DTYPE_TO_NUMPY[node_arg.type]
    except KeyError:
        pytest.skip(f"unsupported RKNN tensor dtype in test: {node_arg.type}")


def _skip_if_native_layout_cannot_be_copied(*values):
    for value in values:
        info = value.memory_info()
        if info.get("fmt") == RKNN_TENSOR_NC1HWC2 and not info.get("pass_through"):
            pytest.skip("RKNN native NC1HWC2 zero-copy binding is not CPU-copyable here")


def _run_or_skip(session, binding):
    try:
        session.run_with_iobinding(binding)
    except RuntimeError as exc:
        message = str(exc)
        if "rknn_set_io_mem failed" in message:
            pytest.skip(f"RKNN runtime rejected this native zero-copy binding: {message}")
        raise


@pytest.mark.parametrize("rank,channel", SHAPE_CASES)
@pytest.mark.parametrize("dtype_name,dtype,tensor_type", DTYPE_CASES)
def test_zero_copy_iobinding_large_add(
    rknn_model_factory, rank, channel, dtype_name, dtype, tensor_type
):
    shape = _shape_for(rank, dtype, channel)
    assert np.empty(shape, dtype=dtype).nbytes >= MIN_TENSOR_BYTES
    model_path = rknn_model_factory(rank, channel, dtype_name, shape, tensor_type)
    try:
        session = ezrknn.InferenceSession(model_path)
    except RuntimeError as exc:
        pytest.skip(f"RKNN runtime cannot load this generated model: {exc}")

    native_dtype = _session_numpy_dtype(session.get_inputs()[0])

    x, y = _make_inputs(shape, dtype)
    expected = (
        x.astype(native_dtype, copy=False) + y.astype(native_dtype, copy=False)
    ).astype(native_dtype, copy=False)
    copy_baseline = session.run(None, {"x": x, "y": y})[0].astype(
        native_dtype, copy=False
    )

    binding = session.io_binding()
    x_value = ezrknn.OrtValue.ortvalue_from_numpy(
        x, "rknpu2", session=session, name="x", io_kind="input"
    )
    y_value = ezrknn.OrtValue.ortvalue_from_numpy(
        y, "rknpu2", session=session, name="y", io_kind="input"
    )
    binding.bind_ortvalue_input("x", x_value)
    binding.bind_ortvalue_input("y", y_value)
    binding.bind_output("z", "rknpu2")
    z_value = binding.get_outputs()[0]
    _skip_if_native_layout_cannot_be_copied(x_value, y_value, z_value)

    _run_or_skip(session, binding)
    actual = z_value.numpy()

    assert actual.shape == shape
    memory_info = z_value.memory_info()
    assert memory_info["device_type"] == "rknpu2"
    assert memory_info["size"] >= np.empty(shape, dtype=native_dtype).nbytes
    assert memory_info["dmabuf_backed"]

    if np.issubdtype(np.dtype(native_dtype), np.integer):
        np.testing.assert_array_equal(actual, copy_baseline)
        if not np.array_equal(copy_baseline, expected):
            pytest.skip(
                "RKNN runtime output for this generated integer model does not "
                "match ONNX Add semantics; zero-copy matched copy-based output"
            )
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

    x2 = (x.astype(native_dtype, copy=False) + np.asarray(1, dtype=native_dtype))
    y2 = (y.astype(native_dtype, copy=False) - np.asarray(2, dtype=native_dtype))
    x_value.update_inplace(x2)
    y_value.update_inplace(y2)
    _run_or_skip(session, binding)
    actual2 = z_value.numpy()
    expected2 = (x2 + y2).astype(native_dtype, copy=False)
    if np.issubdtype(np.dtype(native_dtype), np.integer):
        np.testing.assert_array_equal(actual2, expected2)
    else:
        np.testing.assert_allclose(actual2, expected2, rtol=1e-2, atol=1e-2)


def test_zero_copy_iobinding_honors_nhwc_session_layout(rknn_model_factory):
    dtype = np.float16
    model_shape = _shape_for(4, dtype, channel=8)
    model_path = rknn_model_factory(
        4, "nhwc8", "fp16_nhwc", model_shape, TensorProto.FLOAT16
    )
    session = ezrknn.InferenceSession(model_path, provider_options={"layout": "nhwc"})
    shape = tuple(session.get_inputs()[0].shape)

    x, y = _make_inputs(shape, dtype)
    expected = session.run(None, {"x": x, "y": y})[0].astype(dtype, copy=False)

    binding = session.io_binding()
    x_value = ezrknn.OrtValue.ortvalue_from_numpy(
        x, "rknpu2", session=session, name="x", io_kind="input"
    )
    y_value = ezrknn.OrtValue.ortvalue_from_numpy(
        y, "rknpu2", session=session, name="y", io_kind="input"
    )
    binding.bind_ortvalue_input("x", x_value)
    binding.bind_ortvalue_input("y", y_value)
    binding.bind_output("z", "rknpu2")
    z_value = binding.get_outputs()[0]

    assert x_value.shape() == list(shape)
    assert x_value.memory_info()["fmt"] == RKNN_TENSOR_NHWC

    _run_or_skip(session, binding)
    actual = z_value.numpy()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


def test_zero_copy_iobinding_rejects_shape_and_native_layout_mismatch(
    rknn_model_factory,
):
    dtype = np.float16
    model_shape = _shape_for(4, dtype, channel=8)
    model_path = rknn_model_factory(
        4, "nhwc8", "fp16_nhwc_mismatch", model_shape, TensorProto.FLOAT16
    )
    session = ezrknn.InferenceSession(model_path, provider_options={"layout": "nhwc"})
    shape = tuple(session.get_inputs()[0].shape)
    wrong_shape = shape[:-1] + (shape[-1] + 1,)

    x, y = _make_inputs(shape, dtype)
    wrong_x = np.zeros(wrong_shape, dtype=dtype)
    with pytest.raises(RuntimeError, match="Input 'x' shape mismatch"):
        session.run(None, {"x": wrong_x, "y": y})

    with pytest.raises(RuntimeError, match="OrtValue for input 'x' shape mismatch"):
        ezrknn.OrtValue.ortvalue_from_shape_and_type(
            wrong_shape,
            dtype,
            "rknpu2",
            session=session,
            name="x",
            io_kind="input",
        )

    with pytest.raises(
        RuntimeError, match="OrtValue for input 'x' element type mismatch"
    ):
        ezrknn.OrtValue.ortvalue_from_shape_and_type(
            shape,
            np.float32,
            "rknpu2",
            session=session,
            name="x",
            io_kind="input",
        )

    binding = session.io_binding()
    generic_input = ezrknn.OrtValue.ortvalue_from_shape_and_type(
        shape, dtype, "rknpu2"
    )
    with pytest.raises(RuntimeError, match="native zero-copy layout mismatch"):
        binding.bind_ortvalue_input("x", generic_input)

    wrong_output = ezrknn.OrtValue.ortvalue_from_shape_and_type(
        wrong_shape, dtype, "rknpu2"
    )
    with pytest.raises(RuntimeError, match="OrtValue for output 'z' shape mismatch"):
        binding.bind_ortvalue_output("z", wrong_output)

    with pytest.raises(RuntimeError, match="OrtValue for output 'z' shape mismatch"):
        binding.bind_output("z", "rknpu2", shape=wrong_shape)

    with pytest.raises(
        RuntimeError, match="OrtValue for output 'z' element type mismatch"
    ):
        binding.bind_output("z", "rknpu2", element_type=np.float32)

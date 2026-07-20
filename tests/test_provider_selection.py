import numpy as np
import pytest

import ztu_somemodelruntime_ez_rknn_async as ezrknn


def test_provider_option_helpers_keep_backends_separate():
    assert ezrknn.make_rknpu2_provider_options is ezrknn.make_provider_options
    assert ezrknn.make_rknpu3_provider_options() == {
        "priority": 0,
        "run_timeout_ms": 0,
        "run_core_mask": "auto",
    }
    assert ezrknn.make_rknpu3_provider_options(
        runtime_library="librknn3_api.so",
        weight_path="model.weight",
        run_core_mask=3,
    ) == {
        "runtime_library": "librknn3_api.so",
        "weight_path": "model.weight",
        "priority": 0,
        "run_timeout_ms": 0,
        "run_core_mask": 3,
    }


@pytest.mark.parametrize(
    "model,provider,detected",
    [
        (b"RKNXdummy", "RKNPU2ExecutionProvider", "RKNPU3"),
        (b"RKNNdummy", "RKNPU3ExecutionProvider", "RKNPU2"),
    ],
)
def test_explicit_provider_rejects_the_other_model_header(
    model, provider, detected
):
    with pytest.raises(
        RuntimeError,
        match=f"Provider/model mismatch: requested .* but the model header is {detected}",
    ):
        ezrknn.InferenceSession(model, providers=[provider])


def test_unknown_provider_is_rejected_before_runtime_loading():
    with pytest.raises(RuntimeError, match="Unsupported execution provider"):
        ezrknn.InferenceSession(b"RKNXdummy", providers=["CPUExecutionProvider"])


def test_rknpu3_value_requires_rknpu3_session_context():
    with pytest.raises(RuntimeError, match="standalone RKNPU3 OrtValue"):
        ezrknn.OrtValue.ortvalue_from_shape_and_type(
            [1, 2, 3], np.float16, "rknpu3"
        )

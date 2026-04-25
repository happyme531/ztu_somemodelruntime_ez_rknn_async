from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .options import RknnProviderOptions, make_provider_options

__version__: str


class NodeArg:
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> List[int]: ...
    @property
    def type(self) -> str: ...

class ModelMetadata:
    @property
    def custom_metadata_map(self) -> dict[str, str]: ...

class OrtValue:
    @staticmethod
    def ortvalue_from_numpy(
        arr: Any,
        device_type: str = "cpu",
        device_id: int = 0,
        *,
        name: Optional[str] = None,
        session: Optional["InferenceSession"] = None,
        io_kind: Optional[str] = None,
        layout: str = "onnx",
        sync: bool = True,
    ) -> "OrtValue": ...

    @staticmethod
    def ortvalue_from_shape_and_type(
        shape: Sequence[int],
        element_type: Any,
        device_type: str = "cpu",
        device_id: int = 0,
        *,
        name: Optional[str] = None,
        session: Optional["InferenceSession"] = None,
        io_kind: Optional[str] = None,
        layout: str = "native",
        mem_flags: Union[str, int, None] = "cacheable",
    ) -> "OrtValue": ...

    @staticmethod
    def ortvalue_from_dmabuf(
        fd: int,
        shape: Sequence[int],
        element_type: Any,
        *,
        size: int,
        offset: int = 0,
        virt_addr: Optional[int] = None,
        session: Optional["InferenceSession"] = None,
        name: Optional[str] = None,
        io_kind: Optional[str] = None,
        layout: str = "native",
    ) -> "OrtValue": ...

    def numpy(self) -> Any: ...
    def update_inplace(self, np_arr: Any) -> None: ...
    def data_ptr(self) -> int: ...
    def device_name(self) -> str: ...
    def device_type(self) -> str: ...
    def device_id(self) -> int: ...
    def shape(self) -> List[int]: ...
    def element_type(self) -> str: ...
    def is_tensor(self) -> bool: ...
    def has_value(self) -> bool: ...
    def sync_to_device(self) -> None: ...
    def sync_from_device(self) -> None: ...
    def memory_info(self) -> Dict[str, Any]: ...

class SessionIOBinding:
    def bind_cpu_input(self, name: str, arr_on_cpu: Any) -> None: ...
    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: Any,
        shape: Sequence[int],
        buffer_ptr: int,
    ) -> None: ...
    def bind_ortvalue_input(self, name: str, ortvalue: OrtValue) -> None: ...
    def bind_ortvalue_output(self, name: str, ortvalue: OrtValue) -> None: ...
    def bind_output(
        self,
        name: str,
        device_type: str = "cpu",
        device_id: int = 0,
        element_type: Any = None,
        shape: Optional[Sequence[int]] = None,
        buffer_ptr: Optional[int] = None,
        *,
        layout: str = "native",
        mem_flags: Union[str, int, None] = "cacheable",
    ) -> None: ...
    def get_outputs(self) -> List[OrtValue]: ...
    def copy_outputs_to_cpu(self) -> List[Any]: ...
    def clear_binding_inputs(self) -> None: ...
    def clear_binding_outputs(self) -> None: ...
    def synchronize_inputs(self) -> None: ...
    def synchronize_outputs(self) -> None: ...

IOBinding = SessionIOBinding


class InferenceSession:
    def __init__(
        self,
        path_or_bytes: Any,
        sess_options: Any = ...,
        providers: Any = ...,
        provider_options: Optional[
            Union[Sequence[RknnProviderOptions], RknnProviderOptions]
        ] = ...,
        **kwargs: Any,
    ) -> None: ...

    def run(
        self,
        output_names: Optional[Sequence[str]],
        input_feed: Any,
        run_options: Any = ...,
    ) -> List[Any]: ...

    def run_async(
        self,
        output_names: Optional[Sequence[str]],
        input_feed: Any,
        callback: Callable[[List[Any], Any, Optional[str]], None],
        user_data: Any = ...,
        run_options: Any = ...,
    ) -> None: ...

    def run_pipeline(self, input_feed: Any, depth: int = 3, reset: bool = False) -> Any: ...

    def io_binding(self) -> SessionIOBinding: ...
    def run_with_iobinding(
        self, io_binding: SessionIOBinding, run_options: Any = ...
    ) -> None: ...

    def get_inputs(self) -> List[NodeArg]: ...
    def get_modelmeta(self) -> ModelMetadata: ...
    def get_outputs(self) -> List[NodeArg]: ...
    def get_native_inputs(self) -> List[NodeArg]: ...
    def get_native_outputs(self) -> List[NodeArg]: ...

    @property
    def input_names(self) -> List[str]: ...
    @property
    def output_names(self) -> List[str]: ...


__all__: List[str]

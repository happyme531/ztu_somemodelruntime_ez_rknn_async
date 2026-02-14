from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .options import RknnProviderOptions, make_provider_options

__version__: str


class NodeArg:
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> List[int]: ...
    @property
    def type(self) -> str: ...


class InferenceSession:
    def __init__(
        self,
        path_or_bytes: Any,
        sess_options: Any = ...,
        providers: Any = ...,
        provider_options: Optional[
            Sequence[RknnProviderOptions] | RknnProviderOptions
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

    def get_inputs(self) -> List[NodeArg]: ...
    def get_outputs(self) -> List[NodeArg]: ...

    @property
    def input_names(self) -> List[str]: ...
    @property
    def output_names(self) -> List[str]: ...


__all__: list[str]

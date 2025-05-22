from abc import abstractmethod
from enum import IntEnum
from typing import Any, Dict, TypeVar, Type, List, Callable

from pydantic import BaseModel

from src.utils import file
from src.utils.marshal.protocol import Marshaller
from src.utils.marshal.yaml import DEFAULT_MARSHALLER
from src.utils.tool_functions import init_instance_by_config

T = TypeVar("T", bound="Node")


class Node(BaseModel):
    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...

    @classmethod
    def from_gallery(
            cls: Type[T],
            path: str,
            marshaller: Marshaller = DEFAULT_MARSHALLER,
            **kwargs) -> T:
        fp = file.load_from_gallery(path)
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], **kwargs) -> T:
        return init_instance_by_config(data)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        return self.model_dump()

    def to_gallery(self, path: str, **kwargs):
        ...

    def with_dependencies(self, denpendencies: List[str]) -> T:
        if isinstance(self, RuntimeNode):
            self.dependencies = denpendencies
        else:
            runtime_node = RuntimeNode()
            runtime_node.bound = self
            runtime_node.dependencies = denpendencies
            return runtime_node

    def with_retry(self, max_retry_cnt: int) -> T:
        if isinstance(self, RuntimeNode):
            self.retry_max_cnt = max_retry_cnt
        else:
            runtime_node = RuntimeNode()
            runtime_node.bound = self
            return runtime_node

    def with_check_in(self, **kwargs) -> T:
        ...

    def __or__(self, other):
        ...

    def __ror__(self, other):
        ...


class RuntimeStatus(IntEnum):
    READY: 1
    RUNNING: 2
    WAIT: 3
    DONE: 4
    ERROR: 5


class RuntimeNode(Node):
    bound: Node
    runtime_name: str
    dependencies: List[str]
    check_in_func: Callable[[Any],bool]
    retry_max_cnt: int = 1
    runtime_status: RuntimeStatus
    timeout_seconds: int

    def run(self, **kwargs):
        self._run_with_runtime_info(**kwargs)

    def _run_with_runtime_info(self,**kwargs):
        if self.check_in_func:
            for _ in range(self.retry_max_cnt):
                self.bound.run()



    def update_status(self, new_status: RuntimeStatus):
        self.runtime_status = new_status
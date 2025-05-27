import threading
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Dict, Any, Type, List
from src.data.base import BaseData
from src.base_module import BaseModule, RuntimeModule, ModuleLike
from _del.executor.queue import ThreadSafeQueue
from _del import file
from src.utils import Marshaller
from _del.marshal.yaml import DEFAULT_MARSHALLER


class BaseExecutor(ABC):

    @abstractmethod
    def run(self, data: BaseData):
        ...

    @abstractmethod
    def run_batch(self, datas: List[BaseData]):
        ...

    @abstractmethod
    def add_module(self, module: ModuleLike):
        ...


class MultiStrategyExecutor(BaseExecutor):

    def __init__(self,
                 modules: list[ModuleLike],
                 enable_parallel: bool = True,
                 thread_pool_max_workers: int = 10,
                 timeout_seconds: int = 120
                 ):
        self.enable_parallel = enable_parallel
        self.lock = threading.Lock()
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(thread_pool_max_workers)
        self.modules = self.prepare_modules(modules)
        self._queue = ThreadSafeQueue(self.modules, self.lock)
        self.timeout_seconds = timeout_seconds

    def prepare_modules(self, modules: list[ModuleLike]) -> list[RuntimeModule]:
        ...

    def run(self, data: BaseData):
        while True:
            node = self._queue.get_next_ready()
            if not node and self._queue.get_ready_wait_cnt() < 1:
                break
            self._run_module(node, data)

    def _run_module(self, node: BaseModule, data: BaseData):
        data_copy = deepcopy(data)
        if self.enable_parallel:
            future = self.thread_pool.submit(node.run, **{"data": data_copy})
            future.add_done_callback(partial(default_done_call_back, self._queue, node, self.lock, data, data_copy))
        else:
            node.run(**{"data": data_copy})
            data.update(data_copy)

    def __del__(self):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    @classmethod
    def from_gallery(
            cls: Type[T],
            path: str,
            marshaller: Marshaller = DEFAULT_MARSHALLER,
            **kwargs) -> T:
        fp = file.load_from_gallery(path)
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    @classmethod
    def from_dict(
            cls: Type[T],
            data: Dict[str, Any],
            **kwargs) -> T:
        nodes = []
        nodes_content: List[Dict] = data.get("nodes", [{}])
        for node_data in nodes_content:
            if "class" in node_data:
                init_params = node_data.get("params", {})
                node: BaseModule = BaseModule.from_dict(init_params)
            elif "path" in node_data:
                node = BaseModule.from_gallery(node_data.get("path"))
            else:
                raise
            if "dependencies" in node_data:
                node = node.with_dependencies(node_data["dependencies"])
            if "max_retry_cnt" in node_data:
                node = node.with_retry(node_data["max_retry_cnt"])
            if "check_in" in node_data:
                node = node.with_check_in(**node_data["check_in"])
            nodes.append(node)
        return cls(nodes=nodes)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        pass

    def to_gallery(self, path: str, **kwargs):
        pass











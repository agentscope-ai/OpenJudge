import threading
from concurrent.futures import ThreadPoolExecutor,Future
from copy import deepcopy
from functools import partial
from typing import Dict, Any, Type, List
from typing_extensions import TypeVar
from src.data.base import BaseData
from src.pipeline.node.base import Node, RuntimeNode, RuntimeStatus
from src.pipeline.queue import ThreadSafeDict
from src.utils import file, tool_functions
from threading import Lock

from src.utils.marshal import Marshaller
from src.utils.marshal.yaml import DEFAULT_MARSHALLER

T = TypeVar("T",bound="PipelineBase")

def default_done_call_back(queue:ThreadSafeDict, node:RuntimeNode, lock: Lock, data:BaseData,data_copy:BaseData,future:Future):
    try:
        result = future.result(timeout=120)
        #update data
        with lock:
            data.update(data_copy)
        #update queue
        node.update_status(RuntimeStatus.DONE)
        queue.update_status()
    except Exception as e:
        node.update_status(RuntimeStatus.ERROR)
        queue.update_status()

class PipelineBase:

    def __init__(self,
                 nodes: list[tuple[str,RuntimeNode]],
                 enable_parallel:bool = True,
                 thread_pool_max_workers: int = 10,
                 timeout_seconds:int = 120
                 ):
        self.enable_parallel = enable_parallel
        self.lock = threading.Lock()
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(thread_pool_max_workers)
        self._queue = ThreadSafeDict(nodes,self.lock)
        self.timeout_seconds = timeout_seconds

    def run(self,data:BaseData):
        while True:
            node = self._queue.get_next_ready()
            if not node and self._queue.get_ready_wait_cnt() < 1:
                break
            self._run_node(node,data)


    def run_batch(self,datas: List[BaseData]):
        for data in datas:
            self.run(data)

    def _run_node(self,node:Node,data:BaseData):
        data_copy = deepcopy(data)
        if self.enable_parallel:
            future = self.thread_pool.submit(node.run,**{"data":data_copy})
            future.add_done_callback(partial(default_done_call_back,self._queue,node,self.lock,data,data_copy))
        else:
            node.run(**{"data":data_copy})
            data.update(data_copy)

    def __del__(self):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    @classmethod
    def from_gallery(
            cls: Type[T],
            path: str,
            marshaller:Marshaller = DEFAULT_MARSHALLER,
            **kwargs) -> T :
        fp = file.load_from_gallery(path)
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    @classmethod
    def from_dict(
            cls: Type[T],
            data: Dict[str,Any],
            **kwargs) ->  T :
        nodes = []
        nodes_content: List[Dict] = data.get("nodes",[{}])
        for node_data in nodes_content:
            if "class" in node_data:
                init_params = node_data.get("params",{})
                node: Node = Node.from_dict(init_params)
            elif "path" in node_data:
                node = Node.from_gallery(node_data.get("path"))
            else:
                raise
            if "dependencies" in node_data:
                node = node.with_dependencies(node_data["dependencies"])
            if "max_retry_cnt" in node_data:
                node = node.with_retry(node_data["max_retry_cnt"])
            if "check_in" in node_data:
                node = node.with_check_in(**node_data["check_in"])
            nodes.append(node)
        return cls(nodes = nodes)


    def to_dict(self,**kwargs) -> Dict[str,Any]:
        pass

    def to_gallery(self,path: str,**kwargs):
        pass











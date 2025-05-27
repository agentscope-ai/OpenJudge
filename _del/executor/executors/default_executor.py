
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
from threading import Lock
from src.data.base import BaseData
from _del.executor.base import BaseExecutor, ExecutorCallBackHandler, RuntimeModule, BaseModule, RuntimeStatus
from _del.executor.callbacks.default_handler import DefaultExecutorCallbackHandler
from _del.executor.queue import ThreadSafeQueue

class DefaultExecutor(BaseExecutor):
    """
    default executor for
    """

    def __init__(self,
                 modules: list[BaseModule] = None,
                 enable_parallel:bool = True,
                 thread_pool_max_workers: int = 10,
                 timeout_seconds:int = 120,
                 callback_handler: ExecutorCallBackHandler = DefaultExecutorCallbackHandler()
                 ):
        self.enable_parallel = enable_parallel
        self.lock = Lock()
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(thread_pool_max_workers)
        self.runtime_modules = []
        self.init_modules(modules)
        self._queue = ThreadSafeQueue(self.runtime_modules, self.lock)
        self.timeout_seconds = timeout_seconds
        self.callback_handler = callback_handler

    def init_modules(self, modules:list[BaseModule]):
        #covert to RuntimeModule
        for module in modules:
            if not isinstance(module,RuntimeModule):
                module = self._covert_to_runtime_module(module)
            self.runtime_modules.append(module)
        # modify module dependencies
        self.handler_dependencies(self.runtime_modules)
        #init runtime status
        for module in self.runtime_modules:
            if (module.runtime_status ==
                    RuntimeStatus.INIT):
                if not module.dependencies:
                    module.runtime_status = RuntimeStatus.READY
                else:
                    module.runtime_status = RuntimeStatus.WAIT


    def handler_dependencies(self,modules:list[BaseModule]):
        ...


    def _check_module(self,module:BaseModule):
        ...

    def _covert_to_runtime_module(self,module:BaseModule):
        return module.with_runtime_name()

    def run(self,data:BaseData):
        while True:
            node = self._queue.get_next_ready()
            if not node and self._queue.get_ready_wait_cnt() < 1:
                break
            self._run_module(node, data)

    def _run_module(self, module:RuntimeModule, data:BaseData):
        # data_copy = deepcopy(data)
        if self.enable_parallel:
            future = self.thread_pool.submit(module.run, **{"data":data})
            future.add_done_callback(partial(self.callback_handler.on_module_done(module,self._queue)))
        else:
            try:
                module.run(data=data)
                self.callback_handler.on_module_done(module,self._queue)
            except Exception as e:
                self.callback_handler.on_module_error(module,self._queue,e)

    def run_batch(self, datas: List[BaseData]):
        for data in datas:
            self.run(data)

    def add_module(self, module: BaseModule):
        if not isinstance(module,RuntimeModule):
            module = self._covert_to_runtime_module(module)
            if not module.dependencies:
                module.runtime_status = RuntimeStatus.READY
            else:
                module.runtime_status = RuntimeStatus.WAIT
            self._queue.push(module)


    def __del__(self):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
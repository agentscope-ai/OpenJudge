from src.base import RuntimeModule, BaseModule
from src.executor.executors.default_executor import DefaultExecutor


class SimpleExecutor(DefaultExecutor):


    def __init__(self, module: BaseModule, enable_parallel: bool = True, thread_pool_max_workers: int = 10,
                 timeout_seconds: int = 120):
        super().__init__([module], enable_parallel, thread_pool_max_workers, timeout_seconds)

from src.data.base import BaseData
from src.executor.base import ExecutorCallBackHandler, RuntimeModule, RuntimeStatus
from src.executor.queue import ThreadSafeQueue
from threading import Lock
from concurrent.futures import Future
class DefaultExecutorCallbackHandler(ExecutorCallBackHandler):

    def on_module_done(self, node: RuntimeModule,**kwargs):
        node.update_status(RuntimeStatus.DONE)
        queue:ThreadSafeQueue = kwargs.pop("queue",None)
        if queue:
            queue.update_status()

    def on_module_error(self, node: RuntimeModule, **kwargs):
        node.update_status(RuntimeStatus.ERROR)


    def on_executor_done(self):
        pass

    def on_executor_error(self):
        pass

    def on_executor_succeed(self):
        pass


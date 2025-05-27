from _del.executor.base import ExecutorCallBackHandler, RuntimeModule, RuntimeStatus
from _del.executor.queue import ThreadSafeQueue


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


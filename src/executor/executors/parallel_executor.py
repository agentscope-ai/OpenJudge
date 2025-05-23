from src.base import RuntimeModule, BaseModule
from src.executor.executors.default_executor import DefaultExecutor


class ParallelExecutor(DefaultExecutor):

    def handler_dependencies(self, modules: list[BaseModule]):
        super().handler_dependencies(modules)
        for module in modules:
            module.dependencies = None
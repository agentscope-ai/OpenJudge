from src.base import RuntimeModule, BaseModule
from src.executor.base import MultiStrategyExecutor
from src.executor.executors.default_executor import DefaultExecutor


class HierarchicalExecutor(DefaultExecutor):

    def handler_dependencies(self, modules: list[BaseModule]):
        super().handler_dependencies(modules)


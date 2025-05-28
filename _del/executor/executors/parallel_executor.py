from rm_gallery.core.base_module import BaseModule
from _del.executor.executors.default_executor import DefaultExecutor


class ParallelExecutor(DefaultExecutor):

    def handler_dependencies(self, modules: list[BaseModule]):
        super().handler_dependencies(modules)
        for module in modules:
            module.dependencies = None
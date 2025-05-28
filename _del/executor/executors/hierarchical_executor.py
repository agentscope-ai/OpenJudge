from rm_gallery.core.base_module import BaseModule
from _del.executor.executors.default_executor import DefaultExecutor


class HierarchicalExecutor(DefaultExecutor):

    def handler_dependencies(self, modules: list[BaseModule]):
        super().handler_dependencies(modules)


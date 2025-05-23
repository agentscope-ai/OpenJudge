from src.base import RuntimeModule, BaseModule
from src.executor.executors.default_executor import DefaultExecutor


class SequenceExecutor(DefaultExecutor):


    def handler_dependencies(self, modules: list[BaseModule]):
        super().handler_dependencies(modules)
        module_names = []
        for module in modules:
            module.dependencies = module_names
            module_names.append(module.runtime_name)

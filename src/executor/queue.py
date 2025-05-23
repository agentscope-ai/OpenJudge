from threading import Lock

from src.executor.base import BaseModule, RuntimeModule, RuntimeStatus


class ThreadSafeQueue:
    def __init__(self, nodes:list[BaseModule], lock:Lock):
        self.nodes = {}
        self.lock = lock
        with self.lock:
            for name,node in nodes:
                self.nodes[name] = node

    def push(self, module:RuntimeModule):
        name,node = module.runtime_name,module
        if name in self.nodes.keys():
            raise Exception
        with self.lock:
            self.nodes[name] = node

    def pop(self,module_runtime_name:str,default_val = None):
        if module_runtime_name  in self.nodes.keys():
            self.nodes.pop(module_runtime_name,default_val)

    def get_next_ready(self):
        for name,node  in self.nodes.items():
            if node.runtime_status == RuntimeStatus.READY:
                return node
        return None

    def get_ready_cnt(self):
        return len([node for _,node in self.nodes.items() if node.runtime_status == RuntimeStatus.READY])

    def get_ready_wait_cnt(self) -> int:
        return len([node for _, node in self.nodes.items() if node.runtime_status in { RuntimeStatus.READY,RuntimeStatus.WAIT}])

    def update_status(self):
        for name ,node in self.nodes:
            if node.dependencies:
                is_ready = True
                for denpend_node_name in node.dependencies:
                    depend_node = self.nodes.get(denpend_node_name,None)
                    if depend_node.runtime_status == RuntimeStatus.ERROR:
                        node.runtime_status = RuntimeStatus.ERROR
                    if depend_node.runtime_status != RuntimeStatus.DONE:
                        is_ready = False
                        break
                if is_ready:
                    node.runtime_status = RuntimeStatus.READY
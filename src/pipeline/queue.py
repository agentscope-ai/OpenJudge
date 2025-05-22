from threading import Lock

from src.pipeline.node.base import Node, RuntimeNode, RuntimeStatus


class ThreadSafeDict:
    def __init__(self,nodes:list[tuple[str,RuntimeNode]],lock:Lock):
        self.nodes = {}
        self.lock = lock
        with self.lock:
            for name,node in nodes:
                self.nodes[name] = node

    def push(self,node_tuple: tuple[(str,RuntimeNode)]):
        name,node = node_tuple[0],node_tuple[1]
        if name in self.nodes.keys():
            raise Exception
        with self.lock:
            self.nodes[name] = node

    def pop(self):
        next_node = self.get_next_ready()
        if not next_node:
            raise Exception
        return next_node

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
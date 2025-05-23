from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, Any

import networkx

from easy_rm.data.data import BaseData
from easy_rm_2.palette.block import PaletteBlock
from rm_gallery.pipeline.base import RewardPipeline


class DagRewardPalette(RewardPipeline):
    nodes: list[tuple[str, PaletteBlock]] = None
    graph = networkx.MultiDiGraph()

    def run(self):
        pass

    def evaluate(self, data: BaseData, thread_pool: ThreadPoolExecutor, **kwargs):
        pass

    def evaluate_batch(self, data: BaseData, thread_pool: ThreadPoolExecutor, **kwargs):
        pass

    def add_block(self, block: tuple[str, PaletteBlock]):
        pass

    def delete_block(self, block_name: str):
        pass

    def connect(self,head_name:str,tail_name:str):
        pass

    @classmethod
    def from_dict(cls, datas: Dict[str, Any]):
        pass

    @classmethod
    def from_template(cls):
        pass

    @classmethod
    def load(cls):
        pass

    @classmethod
    def loads(cls):
        pass


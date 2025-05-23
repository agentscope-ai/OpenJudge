from rm_gallery.pipeline.node.base import Node


class SequenceBlock(Node):

    blocks: list[tuple[str,Node]] = None

    def run(self, **kwargs):
        for _,block in self.blocks:
            block.run(**kwargs)







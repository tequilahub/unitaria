from bequem.nodes import Node

from bequem.nodes.qsvt.qsvt import QSVT


class Inverse(QSVT):
    def __init__(self, A: Node, condition: float, accuracy: float):
        angles = None
        raise NotImplementedError
        super().__init__(A, angles)
        self.condition = condition
        self.accuracy = accuracy

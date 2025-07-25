from bequem.nodes import Node

from bequem.nodes.qsvt.qsvt import QSVT


class AmplitudeAmplificiation(QSVT):
    def __init__(self, A: Node, iterations: int):
        angles = None
        raise NotImplementedError
        super().__init__(A, angles)
        self.A = A
        self.iterations = iterations

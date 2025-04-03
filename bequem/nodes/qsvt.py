from node import Node

class QSVT(Node):

    def __init__(self, A: Node, angles: np.array):
        self.A = A
        self.angles = angles

class Inverse(WrapperNode):

    def __init__(self, A: Node, condition: float, accuracy: float):
        self.A = A
        self.condition = condition
        self.accuracy = accuracy

    def definition(self):
        angles = None
        raise NotImplementedError
        return QSVT(A, angles)


class AmplitudeAmplificiation(WrapperNode):

    def __init__(self, A: Node, iterations: int):
        self.A = A
        self.iterations = iterations

    def definition(self):
        angles = None
        raise NotImplementedError
        return QSVT(A, angles)

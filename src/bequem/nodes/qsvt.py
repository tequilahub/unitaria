from node import Node


class QSVT(Node):

    def __init__(self, A: Node, angles: np.ndarray):
        self.A = A
        self.angles = angles


class Inverse(QSVT):

    def __init__(self, A: Node, condition: float, accuracy: float):
        angles = None
        raise NotImplementedError
        super().__init__(A, angles)
        self.condition = condition
        self.accuracy = accuracy


class AmplitudeAmplificiation(QSVT):

    def __init__(self, A: Node, iterations: int):
        angles = None
        raise NotImplementedError
        super().__init__(A, angles)
        self.A = A
        self.iterations = iterations

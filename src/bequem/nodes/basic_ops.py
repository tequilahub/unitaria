from node import Node


class Mul(Node):

    def __init__(self, A, B):
        assert A.qubits_out().simplify().reduce() == B.qubits_in().simplify().reduce()
        self.A = A
        self.B = B

    def compute(self, input: np.array | None) -> np.array:
        input = A.compute(input)
        input = B.compute(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        circuit.append(A.circuit())
        raise NotImplementedError
        # TODO: Qubit permutation
        circuit.append(B.circuit())

        return circuit

    def qubits_in(self) -> QubitMap:
        self.A.qubits_in() # TODO: Stimmt noch nicht

    def qubits_out(self) -> QubitMap:
        self.B.qubits_out()

    def normalization(self) -> float:
        self.A.normalization() * self.B.normalization()

class Tensor(Node):
    pass

class Adjoint(Node):
    pass

class Scale(Node):

    def __init__(self, A, scale: float=1, remove_efficiency: float=1, scale_absolute=False):
        self.scale = scale
        self.remove_efficiency = remove_efficiency
        self.scale_absolute = scale_absolute

from node import Node


class Mul(Node):

    def __init__(self, children):
        assert len(children) == 2
        # Check that qubits maps match
        self.__init__(children[0], children[1])
            
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
        self.A.qubits_in()

    def qubits_out(self) -> QubitMap:
        self.B.qubits_out()

    def normalization(self) -> float:
        self.A.normalization() * self.B.normalization()

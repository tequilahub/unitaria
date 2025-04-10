from __future__ import annotations
import numpy as np

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.node import Node


class Mul(Node):

    def __init__(self, A: Node, B: Node):
        assert \
            A.qubits_out().simplify().reduce() == \
            B.qubits_in().simplify().reduce()
        self.A = A
        self.B = B

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        circuit.append(self.A.circuit())
        raise NotImplementedError
        # TODO: Qubit permutation
        circuit.append(self.B.circuit())

        return circuit

    def qubits_in(self) -> QubitMap:
        self.A.qubits_in()  # TODO: Stimmt noch nicht

    def qubits_out(self) -> QubitMap:
        self.B.qubits_out()

    def normalization(self) -> float:
        self.A.normalization() * self.B.normalization()


class Tensor(Node):
    pass


class Adjoint(Node):
    pass


class Scale(Node):

    def __init__(self,
                 A: Node,
                 scale: float = 1,
                 remove_efficiency: float = 1,
                 scale_absolute=False):
        self.scale = scale
        self.remove_efficiency = remove_efficiency
        self.scale_absolute = scale_absolute

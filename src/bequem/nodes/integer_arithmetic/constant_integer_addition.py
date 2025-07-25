import numpy as np

from ..node import Node
from bequem.subspace import Subspace
from bequem.circuit import Circuit
from bequem.circuits.arithmetic import const_addition_circuit


class ConstantIntegerAddition(Node):
    bits: int
    constant: int

    def __init__(self, bits: int, constant: int):
        if bits < 1:
            raise ValueError()
        self.bits = bits
        self.constant = constant

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits, "constant": self.constant}

    def _subspace_in(self) -> Subspace:
        return Subspace(self.bits, 2)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.bits, 2)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, self.constant, axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, -self.constant, axis=-1)

    def _circuit(self) -> Circuit:
        circuit = const_addition_circuit(list(reversed(range(self.bits))), self.constant, [self.bits, self.bits + 1])
        circuit.n_qubits = self.subspace_in.total_qubits

        return Circuit(circuit)

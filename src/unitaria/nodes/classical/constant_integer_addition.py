import numpy as np

from ..node import Node
from .classical import Classical
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
from unitaria.circuits.arithmetic import const_addition_circuit


class ConstantIntegerAddition(Classical):
    """
    Node implementing the (wrapping) addition of a constant to an integer.

    :param bits:
        The size of the quantum state. The addition is performed modulo ``2 ** bits``.
    :param constant:
        The contant that should be added. Has to be positive.
    """

    bits: int
    constant: int

    def __init__(self, bits: int, constant: int):
        super().__init__(bits, bits)
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

    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        return (input + self.constant) % 2**self.bits

    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        return (input - self.constant) % 2**self.bits

    def _circuit(self) -> Circuit:
        circuit = const_addition_circuit(range(self.bits), self.constant, [self.bits, self.bits + 1])
        circuit.n_qubits = self.subspace_in.total_qubits

        return Circuit(circuit)

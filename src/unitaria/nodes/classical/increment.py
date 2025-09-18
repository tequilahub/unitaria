import numpy as np
import tequila as tq

from ..node import Node
from .classical import Classical
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
from unitaria.circuits.arithmetic import increment_circuit_single_ancilla


class Increment(Classical):
    """
    Node implementing the (wrapping) increment of an integer.

    :param bits:
        The size of the quantum state. The increment is performed modulo ``2 ** bits``.
    """

    bits: int

    def __init__(self, bits: int):
        super().__init__(bits, bits)
        self.bits = bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits}

    def _subspace_in(self) -> Subspace:
        if self.bits <= 3:
            return Subspace(self.bits)
        else:
            return Subspace(self.bits, 1)

    def _subspace_out(self) -> Subspace:
        if self.bits <= 3:
            return Subspace(self.bits)
        else:
            return Subspace(self.bits, 1)

    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        return (input + 1) % 2**self.bits

    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        return (input - 1) % 2**self.bits

    def _circuit(self) -> Circuit:
        if self.bits <= 3:
            circuit = Circuit()
            for i in reversed(range(self.bits)):
                circuit.tq_circuit += tq.gates.X(target=i, control=list(range(i)))
            return circuit
        else:
            circuit = increment_circuit_single_ancilla(target=range(self.bits), ancilla=self.bits)
            return Circuit(circuit)

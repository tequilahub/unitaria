from typing import Sequence

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

    def __init__(self, *args, bits: int = None):
        if len(args) > 0 or bits is None:
            raise TypeError("Increment constructor requires bits=... as a keyword argument.")
        super().__init__(bits, bits)
        self.bits = bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits}

    def _subspace_in(self) -> Subspace:
        return Subspace(bits=self.bits)

    def _subspace_out(self) -> Subspace:
        return Subspace(bits=self.bits)

    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        return (input + 1) % 2**self.bits

    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        return (input - 1) % 2**self.bits

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, 1, axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, -1, axis=-1)

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        if self.bits <= 3:
            circuit = Circuit()
            for i in reversed(range(self.bits)):
                circuit += tq.gates.X(target=target[i], control=target[:i])
            return circuit
        else:
            circuit = increment_circuit_single_ancilla(target=target, ancilla=borrowed_ancillae[0])
            return Circuit(circuit)

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 1 if self.bits > 3 else 0

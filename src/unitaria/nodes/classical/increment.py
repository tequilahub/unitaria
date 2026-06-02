from typing import Sequence

import numpy as np
import tequila as tq

from unitaria.nodes.node import Node
from unitaria.nodes.classical.classical import Classical
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

    def __init__(self, *, bits: int = None):
        if bits is None:
            raise TypeError("Increment constructor requires bits=... as a keyword argument.")
        super().__init__(bits, bits)
        self.bits = bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits}

    def _subspace_in(self) -> Subspace:
        return Subspace("#" * self.bits)

    def _subspace_out(self) -> Subspace:
        return Subspace("#" * self.bits)

    def is_guaranteed_unitary(self) -> bool:
        return True

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
            second_ancilla = None
            if len(borrowed_ancillae) > 1:
                second_ancilla = borrowed_ancillae[1]
            if len(clean_ancillae) > 0:
                second_ancilla = clean_ancillae[0]
            circuit = increment_circuit_single_ancilla(
                target=target, ancilla=borrowed_ancillae[0], second_ancilla=second_ancilla
            )
            return Circuit(circuit)

    def _controlled_circuit(
        self, control: int, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        if self.bits <= 2:
            circuit = Circuit()
            for i in reversed(range(self.bits)):
                circuit += tq.gates.X(target=target[i], control=list(target[:i]) + [control])
            return circuit
        else:
            second_ancilla = None
            if len(borrowed_ancillae) > 1:
                second_ancilla = borrowed_ancillae[1]
            if len(clean_ancillae) > 0:
                second_ancilla = clean_ancillae[0]
            circuit = increment_circuit_single_ancilla(
                target=[control] + list(target), ancilla=borrowed_ancillae[0], second_ancilla=second_ancilla
            )
            circuit += tq.gates.X(control)
            return Circuit(circuit)

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 1 if self.bits > 2 else 0

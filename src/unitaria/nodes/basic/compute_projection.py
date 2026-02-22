from __future__ import annotations

from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class SubspaceCircuit(Node):
    """
    Node, which computes whether a vector is in a given subspace.

    This is mostly used internally, for example for the `unitaria.nodes.Mul`
    node. The result of the check is stored in an additional qubit, for which
    `SubspaceCircuit.subspace_in` and `SubspaceCircuit.subspace_out` are set
    to zero.

    :param subspace:
        The subspace which the vector should be in.
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace):
        super().__init__(subspace.dimension, subspace.dimension)
        self.subspace = subspace

    def children(self) -> list[Node]:
        return []

    def _subspace_in(self) -> Subspace:
        return Subspace(registers=self.subspace.registers, zero_qubits=1)

    def _subspace_out(self) -> Subspace:
        return Subspace(registers=self.subspace.registers, zero_qubits=1)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        return self.subspace.circuit(target[:-1], target[-1], clean_ancillae)

    def clean_ancilla_count(self) -> int:
        return self.subspace.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return 0

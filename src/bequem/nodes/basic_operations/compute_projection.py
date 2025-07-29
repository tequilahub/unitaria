from __future__ import annotations
import numpy as np

from bequem.circuit import Circuit
from bequem.subspace import Subspace
from bequem.nodes.node import Node


class ComputeProjection(Node):
    """
    Node, which computes wether a vector is in a given subspace.

    This is mostly used internally, for example for the `bequem.nodes.Mul`
    node. The result of the check is stored in an additional qubit, for which
    `ComputeProjection.subspace_in` and `ComputeProjection.subspace_out` are set
    to zero.

    :ivar subspace:
        The subspace which the vector should be in.
    """

    def __init__(self, subspace: Subspace):
        self.subspace = subspace

    def children(self) -> list[Node]:
        return []

    def _subspace_in(self) -> Subspace:
        # TODO: most of the extra zeros are actually ancillas
        return Subspace(
            self.subspace.registers,
            self.circuit.tq_circuit.n_qubits - self.subspace.total_qubits,
        )

    def _subspace_out(self) -> Subspace:
        return Subspace(
            self.subspace.registers,
            self.circuit.tq_circuit.n_qubits - self.subspace.total_qubits,
        )

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(self) -> Circuit:
        return self.subspace.circuit()

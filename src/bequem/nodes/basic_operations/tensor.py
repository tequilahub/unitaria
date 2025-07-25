from __future__ import annotations
import numpy as np

from bequem.circuit import Circuit
from bequem.subspace.subspace import Subspace
from bequem.nodes.node import Node


class Tensor(Node):
    """
    Node representing the tensor product of two other nodes

    The order of operations is such that ``A`` corresponds to the lower
    significant digits of the index, i.e.

    >>> from bequem.nodes import Tensor, Identity, Increment
    >>> from bequem.qubit_map import QubitMap
    >>> import numpy as np
    >>> Tensor(Increment(1), Identity(QubitMap(1))).compute(np.eye(4))
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        """
        :param A:
            The first factor
        :param B:
            The second factor
        """
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.A.subspace_in.dimension])
        input = self.A.compute(input)
        input = input.reshape(batch_shape + [self.B.subspace_in.dimension, self.A.subspace_out.dimension])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.subspace_in.dimension])
        input = self.B.compute(input)
        input = input.reshape(batch_shape + [self.A.subspace_out.dimension, self.B.subspace_out.dimension])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.A.subspace_out.dimension])
        input = self.A.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.B.subspace_out.dimension, self.A.subspace_in.dimension])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.subspace_out.dimension])
        input = self.B.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.A.subspace_in.dimension, self.B.subspace_in.dimension])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def _circuit(self) -> Circuit:
        circuit = Circuit()

        circuit_A = self.A.circuit.tq_circuit
        circuit.tq_circuit += circuit_A
        qubit_map_B = dict([(i, i + self.A.subspace_in.total_qubits) for i in range(self.B.subspace_in.total_qubits)])
        circuit_B = self.B.circuit.tq_circuit.map_qubits(qubit_map_B)
        circuit.tq_circuit += circuit_B

        circuit.tq_circuit.n_qubits = self.A.subspace_in.total_qubits + self.B.subspace_in.total_qubits

        return circuit

    def _subspace_in(self) -> Subspace:
        subspace_A = self.A.subspace_in
        subspace_B = self.B.subspace_in
        return Subspace(
            subspace_A.registers + subspace_B.registers,
        )

    def _subspace_out(self) -> Subspace:
        subspace_A = self.A.subspace_out
        subspace_B = self.B.subspace_out
        return Subspace(
            subspace_A.registers + subspace_B.registers,
        )

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization


Node.__and__ = lambda A, B: Tensor(A, B)

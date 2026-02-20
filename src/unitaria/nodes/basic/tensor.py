from __future__ import annotations
from typing import Sequence
import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Tensor(Node):
    """
    Node representing the tensor product of two other nodes

    The order of operations is such that ``A`` corresponds to the lower
    significant digits of the index, i.e.

    >>> from unitaria.nodes import Tensor, Identity, Increment
    >>> import numpy as np
    >>> Tensor(Increment(1), Identity(1)).toarray().real
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    The and operator for ``Node`` is overloaded to be the tensor product,
    i.e. you can equivalently write

    >>> from unitaria.nodes import Tensor, Identity, Increment
    >>> import numpy as np
    >>> (Increment(1) & Identity(1)).toarray().real
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    :param A:
        The first factor
    :param B:
        The second factor
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        super().__init__(A.dimension_in * B.dimension_in, A.dimension_out * B.dimension_out)
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.A.dimension_in])
        input = self.A.compute(input)
        input = input.reshape(batch_shape + [self.B.dimension_in, self.A.dimension_out])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.dimension_in])
        input = self.B.compute(input)
        input = input.reshape(batch_shape + [self.A.dimension_out, self.B.dimension_out])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.A.dimension_out])
        input = self.A.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.B.dimension_out, self.A.dimension_in])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.dimension_out])
        input = self.B.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.A.dimension_in, self.B.dimension_in])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        # TODO: Optionally optimize for qubit count instead of depth?
        circuit = Circuit()
        circuit += self.A.circuit(
            target[: self.A.target_qubit_count()],
            clean_ancillae[: self.A.clean_ancilla_count()],
            borrowed_ancillae[: self.A.borrowed_ancilla_count()],
        )
        circuit += self.B.circuit(
            target[self.A.target_qubit_count() :],
            clean_ancillae[self.A.clean_ancilla_count() :],
            borrowed_ancillae[self.A.borrowed_ancilla_count() :],
        )
        return circuit

    def _subspace_in(self) -> Subspace:
        subspace_A = self.A.subspace_in
        subspace_B = self.B.subspace_in
        return subspace_A & subspace_B

    def _subspace_out(self) -> Subspace:
        subspace_A = self.A.subspace_out
        subspace_B = self.B.subspace_out
        return subspace_A & subspace_B

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count() + self.B.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count() + self.B.borrowed_ancilla_count()


Node.__and__ = lambda A, B: Tensor(A, B)

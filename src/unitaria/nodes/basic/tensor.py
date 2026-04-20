from __future__ import annotations
from typing import Sequence
import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Tensor(Node):
    """
    Node representing the tensor (Kronecker) product of two other nodes, consistent with numpy's np.kron(A, B).

    The order of operations is such that Tensor(A, B) and (A & B) correspond to np.kron(A, B):
    - A acts on the left part, B acts on the right part.
    - In the compute methods, B is applied first, then A, matching the action of np.kron(A, B) on a vector.

    Example (np.kron from numpy):
        >>> import numpy as np
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[0, 5], [6, 7]])
        >>> np.kron(A, B)
        array([[ 0,  5,  0, 10],
               [ 6,  7, 12, 14],
               [ 0, 15,  0, 20],
               [18, 21, 24, 28]])

    Example (unitaria specific):
        >>> import unitaria as ut
        >>> import numpy as np
        >>> ut.Tensor(ut.Increment(bits=1), ut.Identity(bits=1)).toarray().real
        array([[0., 1., 0., 0.],
               [1., 0., 0., 0.],
               [0., 0., 0., 1.],
               [0., 0., 1., 0.]])

    The and operator for `ut.Node` is overloaded to be the tensor product,
    i.e. you can equivalently write

        >>> import unitaria as ut
        >>> import numpy as np
        >>> (ut.Increment(bits=1) & ut.Identity(bits=1)).toarray().real
        array([[0., 1., 0., 0.],
               [1., 0., 0., 0.],
               [0., 0., 0., 1.],
               [0., 0., 1., 0.]])

    :param A:
        The left (second) factor
    :param B:
        The right (first) factor
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
        input = input.reshape([-1, self.B.dimension_in])
        input = self.B.compute(input)
        input = input.reshape(batch_shape + [self.A.dimension_in, self.B.dimension_out])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.A.dimension_in])
        input = self.A.compute(input)
        input = input.reshape(batch_shape + [self.A.dimension_out, self.B.dimension_out])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.B.dimension_out])
        input = self.B.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.A.dimension_out, self.B.dimension_in])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.A.dimension_out])
        input = self.A.compute_adjoint(input)
        input = input.reshape(batch_shape + [self.A.dimension_in, self.B.dimension_in])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()
        circuit += self.B.circuit(
            target[self.B.target_qubit_count() :],
            clean_ancillae[self.B.clean_ancilla_count() :],
            borrowed_ancillae[self.B.borrowed_ancilla_count() :],
        )
        circuit += self.A.circuit(
            target[: self.A.target_qubit_count()],
            clean_ancillae[: self.A.clean_ancilla_count()],
            borrowed_ancillae[: self.A.borrowed_ancilla_count()],
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

    def is_guaranteed_unitary(self) -> bool:
        return self.A.is_guaranteed_unitary() and self.B.is_guaranteed_unitary()

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count() + self.B.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count() + self.B.borrowed_ancilla_count()


Node.__and__ = lambda A, B: Tensor(A, B)

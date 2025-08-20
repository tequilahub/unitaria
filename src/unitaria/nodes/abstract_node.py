from typing import Callable, Sequence

import numpy as np

from unitaria.nodes.node import Node


class AbstractNode(Node):
    """
    Abstract node without circuit implemention.

    This node is meant for the purpose of prototyping or testing. Using
    `subspace_in`, `subspace_out`, `normalization`, or `circuit` on this
    node or any of this parents will likely raise an error.

    You may alternatively use the nodes ``Constant...`` to represent classical
    objects, since these at least have an inefficient circuit implementation.

    :param dimension_in:
        The dimension of the input to the matrix or 1 if the node should
        represent a vector.
    :param dimension_out:
        The dimension of the vector or the output of the matrix.
    :param compute:
        Implementation of `Node.compute`.
    :param compute_adjoint:
        Implementation of `Node.compute_adjoint`.
    """

    def __init__(
        self,
        dimension_in: int,
        dimension_out: int,
        compute: Callable[[np.ndarray], np.ndarray],
        compute_adjoint: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__(dimension_in, dimension_out)
        self.compute_fn = compute
        self.compute_adjoint_fn = compute_adjoint

    def compute(self, input):
        return self.compute_fn(input)

    def compute_adjoint(self, input):
        return self.compute_adjoint_fn(input)

    def _subspace_in(self):
        raise NotImplementedError("Abstract node has no circuit implementation")

    def _subspace_out(self):
        raise NotImplementedError("Abstract node has no circuit implementation")

    def _normalization(self):
        raise NotImplementedError("Abstract node has no circuit implementation")

    def _circuit(self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]):
        raise NotImplementedError("Abstract node has no circuit implementation")

    def clean_ancilla_count(self) -> int:
        return (
            max(
                self.A.subspace_in.clean_ancilla_count(),
                self.A.subspace_out.clean_ancilla_count(),
            )
            + 1
        )

    def borrowed_ancilla_count(self) -> int:
        return 0

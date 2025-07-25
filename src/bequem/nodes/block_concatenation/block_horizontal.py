import numpy as np

from bequem.nodes.node import Node
from bequem.nodes.basic_operations.scale import Scale
from bequem.nodes.basic_operations.adjoint import Adjoint
from bequem.nodes.basic_operations.unsafe_multiplication import UnsafeMul
from bequem.nodes.basic_operations.tensor import Tensor

from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.controlled_operations.block_diagonal import BlockDiagonal
from bequem.nodes.permutation import Permutation
from bequem.nodes.constants.constant_vector import ConstantVector
from bequem.nodes.identity import Identity


class BlockHorizontal(ProxyNode):
    """
    Node for block matrices of the form ``[A B]``

    :ivar A:
        The left block
    :ivar B:
        The right block
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation = Permutation(self.A.subspace_out, self.B.subspace_out)

        A_permuted = Scale(UnsafeMul(self.A, permutation), absolute=True)
        B_permuted = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)

        rotation_out = Tensor(
            Identity(permutation.subspace_out),
            ConstantVector(np.array([self.A.normalization, self.B.normalization])),
        )

        return UnsafeMul(diag, Adjoint(rotation_out))

    def _normalization(self) -> float:
        return np.sqrt(np.abs(self.A.normalization) ** 2 + np.abs(self.B.normalization) ** 2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.subspace_in.dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute(input_A) + self.B.compute(input_B)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate((self.A.compute_adjoint(input), self.B.compute_adjoint(input)), axis=-1)

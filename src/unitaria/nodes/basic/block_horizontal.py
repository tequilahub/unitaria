import numpy as np

from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.basic.scale import Scale
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic.tensor import Tensor
from unitaria.nodes.basic.block_diagonal import BlockDiagonal
from unitaria.nodes.permutation.permutation import Permutation
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.basic.identity import Identity


class BlockHorizontal(ProxyNode):
    """
    Node for block matrices of the form ``[A B]``

    :param A:
        The left block
    :param B:
        The right block
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        if A.dimension_out != B.dimension_out:
            raise ValueError(f"Matrices have different output dimension {A.dimension_out} and {B.dimension_out}")

        super().__init__(A.dimension_in + B.dimension_in, A.dimension_out)
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
            Identity(subspace=permutation.subspace_out),
            ConstantVector(np.array([self.A.normalization, self.B.normalization])),
        )

        return UnsafeMul(diag, Adjoint(rotation_out))

    def _normalization(self) -> float:
        return np.sqrt(np.abs(self.A.normalization) ** 2 + np.abs(self.B.normalization) ** 2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.dimension_in
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute(input_A) + self.B.compute(input_B)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate((self.A.compute_adjoint(input), self.B.compute_adjoint(input)), axis=-1)

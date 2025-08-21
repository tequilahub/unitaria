import numpy as np
from unitaria.nodes.node import Node
from unitaria.nodes.basic_operations.adjoint import Adjoint

from unitaria.nodes.basic_operations.scale import Scale
from unitaria.nodes.basic_operations.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic_operations.tensor import Tensor

from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.controlled_operations.block_diagonal import BlockDiagonal
from unitaria.nodes.permutation import Permutation
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.identity import Identity


class Add(ProxyNode):
    """
    Node for computing the sum of two vectors or matrices.

    Both summands must have matching `Add.dimension_in` and `Add.dimension_out`.

    :param A:
        The first summand
    :param B:
        The second summand
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        if A.subspace_in.dimension != B.subspace_in.dimension:
            raise ValueError(f"dimensions {A.dimension_in} and {B.dimension_in} do not match")
        if A.subspace_out.dimension != B.subspace_out.dimension:
            raise ValueError(f"dimensions {A.dimension_out} and {B.dimension_out} do not match")
        super().__init__(A.dimension_in, A.dimension_out)
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation_in = Permutation(self.A.subspace_in, self.B.subspace_in)
        permutation_out = Permutation(self.A.subspace_out, self.B.subspace_out)

        A_permuted = Scale(
            UnsafeMul(Adjoint(permutation_in), UnsafeMul(self.A, permutation_out)),
            absolute=True,
        )
        B_permuted = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)

        sqrt_A = np.sqrt(np.abs(self.A.normalization))
        sqrt_B = np.sqrt(np.abs(self.B.normalization))
        rotation_in = Tensor(
            Identity(permutation_in.subspace_in),
            ConstantVector(np.array([sqrt_A, sqrt_B])),
        )
        rotation_out = Tensor(
            Identity(permutation_out.subspace_out),
            ConstantVector(np.array([self.A.normalization / sqrt_A, self.B.normalization / sqrt_B])),
        )

        return UnsafeMul(UnsafeMul(rotation_in, diag), Adjoint(rotation_out))

    def _normalization(self) -> float:
        return self.A.normalization + self.B.normalization

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)

Node.__sub__ = lambda A, B: Add(A, Scale(B, -1))

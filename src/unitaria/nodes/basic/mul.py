import numpy as np
from unitaria.nodes.node import Node

from unitaria.nodes.basic.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic.tensor import Tensor
from unitaria.nodes.basic.compute_projection import SubspaceCircuit

from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.permutation.permutation import permute
from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.subspace import Subspace


class Mul(ProxyNode):
    """
    Node for computing the product of two nodes

    The order of operations is, like in matrix multiplication,
    from right to left, i.e. B is applied first so this implements ``A @ B``.
    This requires ``B.dimension_out == A.dimension_in``.

    :param A:
        The left factor
    :param B:
        The right factor
    """

    A: Node
    B: Node
    skip_projection: bool

    def __init__(self, A: Node, B: Node, skip_projection: bool = False):
        if B.dimension_out != A.dimension_in:
            raise ValueError(f"dimensions {B.dimension_out} and {A.dimension_in} do not match")
        super().__init__(B.dimension_in, A.dimension_out)
        self.A = A
        self.B = B
        self.skip_projection = skip_projection

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def parameters(self):
        params = {}
        if self.skip_projection:
            params["skip_projection"] = True
        return params

    def definition(self) -> Node:
        permute_B, permute_A = permute(self.B.subspace_out, self.A.subspace_in)
        A_permuted = UnsafeMul(self.A, Adjoint(permute_A))
        B_permuted = UnsafeMul(permute_B, self.B)
        projection_subspace = B_permuted.subspace_out
        projection_required = not self.skip_projection
        if projection_subspace == Subspace("#" * projection_subspace.total_qubits):
            projection_required = False
        if self.A.is_guaranteed_unitary() or self.B.is_guaranteed_unitary():
            projection_required = False
        if projection_required:
            # TODO: This can be probably be done more correctly
            # once match_nonzero is improved
            if B_permuted.subspace_out.total_qubits < A_permuted.subspace_in.total_qubits:
                projection_subspace = A_permuted.subspace_in
            A_permuted = Tensor(Identity(Subspace("0")), A_permuted)
            B_permuted = Tensor(Identity(Subspace("0")), B_permuted)
            return UnsafeMul(UnsafeMul(A_permuted, SubspaceCircuit(projection_subspace)), B_permuted)
        else:
            return UnsafeMul(A_permuted, B_permuted)

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute(input)
        input = self.A.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute_adjoint(input)
        input = self.B.compute_adjoint(input)
        return input

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization

    def is_guaranteed_unitary(self) -> bool:
        return self.A.is_guaranteed_unitary() and self.B.is_guaranteed_unitary()


Node.__matmul__ = lambda A, B: Mul(A, B)

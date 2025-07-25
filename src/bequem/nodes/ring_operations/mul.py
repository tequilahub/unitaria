import numpy as np
from bequem.nodes.node import Node

from bequem.nodes.basic_operations.unsafe_multiplication import UnsafeMul
from bequem.nodes.basic_operations.tensor import Tensor
from bequem.nodes.basic_operations.compute_projection import ComputeProjection

from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.permutation import Permutation
from bequem.nodes.identity import Identity
from bequem.subspace.subspace import Subspace


class Mul(ProxyNode):
    """
    Node for computing the product of two nodes

    The order of operations is such that the first argument ``A`` is applied
    first, i.e. this implements ``B @ A``.

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """

    A: Node
    B: Node
    skip_projection: bool

    def __init__(self, A: Node, B: Node, skip_projection: bool = False):
        if A.subspace_out.dimension != B.subspace_in.dimension:
            raise ValueError(f"dimensions {A.subspace_out.dimension} and {B.subspace_in.dimension} do not match")
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
        permutation = Permutation(self.A.subspace_out, self.B.subspace_in)
        A_permuted = self.A
        B_permuted = UnsafeMul(permutation, self.B)
        projection_subspace = A_permuted.subspace_out
        projection_required = not self.skip_projection
        if projection_subspace == Subspace(projection_subspace.total_qubits):
            projection_required = False
        if projection_required:
            # TODO: This can be probably be done more correctly
            # once match_nonzero is improved
            if A_permuted.subspace_out.total_qubits < B_permuted.subspace_in.total_qubits:
                projection_subspace = B_permuted.subspace_in
            A_permuted = Tensor(A_permuted, Identity(Subspace(0, 1)))
            B_permuted = Tensor(B_permuted, Identity(Subspace(0, 1)))
            return UnsafeMul(UnsafeMul(A_permuted, ComputeProjection(projection_subspace)), B_permuted)
        else:
            return UnsafeMul(A_permuted, B_permuted)

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization


Node.__matmul__ = lambda A, B: Mul(B, A)

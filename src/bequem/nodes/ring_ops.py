import numpy as np
from bequem.nodes.node import Node
from bequem.nodes.basic_ops import Scale, Adjoint, UnsafeMul, Tensor, ComputeProjection
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.nodes.permutation import Permutation
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.subspace import Subspace


class Add(ProxyNode):
    """
    Node for computing the sum of two nodes

    :ivar A:
        The first summand
    :ivar B:
        The second summand
    """
    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        if A.subspace_in.dimension != B.subspace_in.dimension:
            raise ValueError(f"dimensions {A.subspace_in.dimension} and {B.subspace_in.dimension} do not match")
        if A.subspace_out.dimension != B.subspace_out.dimension:
            raise ValueError(f"dimensions {A.subspace_out.dimension} and {B.subspace_out.dimension} do not match")
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation_in = Permutation(self.A.subspace_in,
                                     self.B.subspace_in)
        permutation_out = Permutation(self.A.subspace_out,
                                      self.B.subspace_out)

        A_permuted = Scale(UnsafeMul(
            Adjoint(permutation_in),
            UnsafeMul(self.A, permutation_out)),
                           absolute=True)
        B_permuted = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)

        sqrt_A = np.sqrt(np.abs(self.A.normalization))
        sqrt_B = np.sqrt(np.abs(self.B.normalization))
        rotation_in = Tensor(Identity(permutation_in.subspace_in),
                             ConstantVector(np.array([sqrt_A, sqrt_B])))
        rotation_out = Tensor(
            Identity(permutation_out.subspace_out),
            ConstantVector(
                np.array([
                    self.A.normalization / sqrt_A,
                    self.B.normalization / sqrt_B
                ])))

        return UnsafeMul(UnsafeMul(rotation_in, diag), Adjoint(rotation_out))

    def _normalization(self) -> float:
        return self.A.normalization + self.B.normalization

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)

Node.__sub__ = lambda A, B: Add(A, Scale(B, -1))


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
        if self.skip_projection:
            A_permuted = self.A
            B_permuted = UnsafeMul(permutation, self.B)
            return UnsafeMul(A_permuted, B_permuted)
        else:
            A_permuted = Tensor(self.A,
                                Identity(Subspace(0, 1)))
            B_permuted = Tensor(UnsafeMul(permutation, self.B),
                                Identity(Subspace(0, 1)))
            # TODO: This can be probably be done more correctly
            # once match_nonzero is improved
            projection_subspace = A_permuted.subspace_out
            if A_permuted.subspace_out.total_qubits < B_permuted.subspace_in.total_qubits:
                projection_subspace = B_permuted.subspace_in
            projection_required = True
            if projection_subspace == Subspace(projection_subspace.total_qubits):
                projection_required = False
            if projection_required:
                return UnsafeMul(
                    UnsafeMul(A_permuted, ComputeProjection(projection_subspace)),
                    B_permuted)
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

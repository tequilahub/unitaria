import numpy as np

from bequem.nodes.basic_ops import Scale, Tensor, Adjoint
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node
from bequem.nodes.mul import Mul
from bequem.nodes.block_diagonal import BlockDiagonal


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
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        A = Scale(self.A, absolute=True)
        B = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A, B)

        sqrt_A = np.sqrt(np.abs(self.A.normalization()))
        sqrt_B = np.sqrt(np.abs(self.B.normalization()))
        rotation_in = Tensor(Identity(self.A.qubits_in()),
                             ConstantVector(np.array([sqrt_A, sqrt_B])))
        rotation_out = Tensor(
            Identity(self.A.qubits_out()),
            ConstantVector(
                np.array([
                    self.A.normalization() / sqrt_A,
                    self.B.normalization() / sqrt_B
                ])))

        return Mul(Mul(rotation_in, diag, skip_projection_check=True), Adjoint(rotation_out), skip_projection_check=True)

    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)


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
        A_permuted = Scale(self.A, absolute=True)
        B_permuted = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)

        rotation_out = Tensor(
            Identity(self.A.qubits_out()),
            ConstantVector(
                np.array([self.A.normalization(),
                          self.B.normalization()])))

        return Mul(diag, Adjoint(rotation_out), skip_projection_check=True)

    def normalization(self) -> float:
        return np.sqrt(
            np.abs(self.A.normalization())**2 +
            np.abs(self.B.normalization())**2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute(input_A) + self.B.compute(input_B)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (self.A.compute_adjoint(input), self.B.compute(input)), axis=-1)


class BlockVertical(ProxyNode):
    """
    Node for block matrices of the form ``[A B]^T``

    :ivar A:
        The top block
    :ivar B:
        The bottom block
    """

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:

        A_permuted = Scale(self.A, absolute=True)
        B_permuted = Scale(self.B, absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)

        rotation_in = Tensor(
            Identity(self.A.qubits_in()),
            ConstantVector(
                np.array([self.A.normalization(),
                          self.B.normalization()])))

        return Mul(rotation_in, diag, skip_projection_check=True)

    def normalization(self) -> float:
        return np.sqrt(
            np.abs(self.A.normalization())**2 +
            np.abs(self.B.normalization())**2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate((self.A.compute(input), self.B.compute(input)),
                              axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute_adjoint(input_A) + self.B.compute_adjoint(
            input_B)


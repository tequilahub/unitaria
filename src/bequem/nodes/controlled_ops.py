import numpy as np

from bequem.qubit_map import QubitMap, Qubit
from bequem.nodes.basic_ops import ModifyControl, Scale, Tensor, Adjoint
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node
from bequem.nodes.mul import Mul


class BlockDiagonal(ProxyNode):

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        assert np.isclose(A.normalization(), B.normalization())

        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self):
        A_controlled = self.A.controlled()
        B_controlled = self.B.controlled()
        qubits_in = _controlled_qubits(A_controlled.qubits_in(), B_controlled.qubits_in())
        qubits_mid = _controlled_qubits(A_controlled.qubits_out(), B_controlled.qubits_in())
        qubits_out = _controlled_qubits(A_controlled.qubits_out(), B_controlled.qubits_out())
        controlled_bits_A = A_controlled.qubits_in().total_qubits - A_controlled.qubits_in().trailing_zeros()
        controlled_bits_B = B_controlled.qubits_in().total_qubits - B_controlled.qubits_in().trailing_zeros()

        A_controlled = ModifyControl(
            A_controlled, max(0, controlled_bits_B - controlled_bits_A), True)
        A_controlled = Mul(
            Mul(
                Identity(qubits_in, A_controlled.qubits_in()),
                A_controlled,
                skip_projection_check=True
            ),
            Identity(A_controlled.qubits_out(), qubits_mid),
            skip_projection_check=True
        )
        B_controlled = ModifyControl(
            B_controlled, max(0, controlled_bits_A - controlled_bits_B), False)
        B_controlled = Mul(
            Mul(
                Identity(qubits_in, B_controlled.qubits_in()),
                B_controlled,
                skip_projection_check=True
            ),
            Identity(B_controlled.qubits_out(), qubits_out),
            skip_projection_check=True
        )

        return Mul(A_controlled, B_controlled, skip_projection_check=True)

    def normalization(self) -> float:
        return self.A.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute(input_A)
        result_B = self.B.compute(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_out().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute_adjoint(input_A)
        result_B = self.B.compute_adjoint(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

def _controlled_qubits(A_controlled: QubitMap, B_controlled: QubitMap) -> QubitMap:
    zeros = max(A_controlled.trailing_zeros(), B_controlled.trailing_zeros())
    A = A_controlled.case_one()
    B = B_controlled.case_one()
    controlled_qubits = max(A.total_qubits, B.total_qubits)
    case_zero = QubitMap(
        A.registers, max(0, controlled_qubits - A.total_qubits))
    case_one = QubitMap(
        B.registers, max(0, controlled_qubits - B.total_qubits))
    return QubitMap([Qubit(case_zero, case_one)], zeros)


Node.__or__ = lambda A, B: BlockDiagonal(A, B)

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


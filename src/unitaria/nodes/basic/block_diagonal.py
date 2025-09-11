import numpy as np

from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.basic.controlled import Controlled
from unitaria.nodes.basic.modify_control import ModifyControl
from unitaria.nodes.basic.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic.projection import Projection
from unitaria.subspace import Subspace, ControlledSubspace


class BlockDiagonal(ProxyNode):
    """
    Node for block matrices of the form ``diag(A, B)``

    The and operator for ``Node`` is overloaded to build a BlockDiagonal node,
    so ``BlockDiagonal(A, B)`` is equivalent to writing ``A | B``.

    :param A:
        The left upper block
    :param B:
        The right lower block
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        assert np.isclose(A.normalization, B.normalization)
        super().__init__(A.dimension_in + B.dimension_in, A.dimension_out + B.dimension_out)

        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self):
        A_controlled = Controlled(self.A)
        B_controlled = Controlled(self.B)
        subspace_in = _controlled_qubits(A_controlled.subspace_in, B_controlled.subspace_in)
        subspace_mid = _controlled_qubits(A_controlled.subspace_out, B_controlled.subspace_in)
        subspace_out = _controlled_qubits(A_controlled.subspace_out, B_controlled.subspace_out)
        controlled_bits_A = A_controlled.subspace_in.total_qubits - A_controlled.subspace_in.trailing_zeros()
        controlled_bits_B = B_controlled.subspace_in.total_qubits - B_controlled.subspace_in.trailing_zeros()

        A_controlled = ModifyControl(A_controlled, max(0, controlled_bits_B - controlled_bits_A), True)
        A_controlled = UnsafeMul(
            UnsafeMul(
                Projection(subspace_in, A_controlled.subspace_in),
                A_controlled,
            ),
            Projection(A_controlled.subspace_out, subspace_mid),
        )
        B_controlled = ModifyControl(B_controlled, max(0, controlled_bits_A - controlled_bits_B), False)
        B_controlled = UnsafeMul(
            UnsafeMul(
                Projection(subspace_mid, B_controlled.subspace_in),
                B_controlled,
            ),
            Projection(B_controlled.subspace_out, subspace_out),
        )

        return UnsafeMul(A_controlled, B_controlled)

    def _normalization(self) -> float:
        return self.A.normalization

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.dimension_in
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute(input_A)
        result_B = self.B.compute(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.dimension_out
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute_adjoint(input_A)
        result_B = self.B.compute_adjoint(input_B)
        return np.concatenate((result_A, result_B), axis=-1)


def _controlled_qubits(A_controlled: Subspace, B_controlled: Subspace) -> Subspace:
    zeros = max(A_controlled.trailing_zeros(), B_controlled.trailing_zeros())
    A = A_controlled.case_one()
    B = B_controlled.case_one()
    controlled_qubits = max(A.total_qubits, B.total_qubits)
    case_zero = Subspace(A.registers, max(0, controlled_qubits - A.total_qubits))
    case_one = Subspace(B.registers, max(0, controlled_qubits - B.total_qubits))
    return Subspace([ControlledSubspace(case_zero, case_one)], zeros)


Node.__or__ = lambda A, B: BlockDiagonal(A, B)

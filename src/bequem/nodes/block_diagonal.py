import numpy as np

from bequem.qubit_map import QubitMap, Qubit
from bequem.nodes.basic_ops import ModifyControl, UnsafeMul
from bequem.nodes.identity import Identity
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node


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
        A_controlled = UnsafeMul(
            UnsafeMul(
                Identity(qubits_in, A_controlled.qubits_in()),
                A_controlled,
            ),
            Identity(A_controlled.qubits_out(), qubits_mid),
        )
        B_controlled = ModifyControl(
            B_controlled, max(0, controlled_bits_A - controlled_bits_B), False)
        B_controlled = UnsafeMul(
            UnsafeMul(
                Identity(qubits_in, B_controlled.qubits_in()),
                B_controlled,
            ),
            Identity(B_controlled.qubits_out(), qubits_out),
        )

        return UnsafeMul(A_controlled, B_controlled)

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

from __future__ import annotations
import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap, Qubit
from bequem.nodes.node import Node


class UnsafeMul(Node):
    """
    Node for chaining the circuits of two nodes

    This is mostly for internal usage. To properly multiply two matrices use
    :py:class:`~bequem.nodes.prox_node.Mul` instead. The order of operations is
    such that the first argument ``A`` is applied first.

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """
    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        """
        The order of operations is such that the first argument ``A`` is applied
        first.

        :ivar A:
            The first factor
        :ivar B:
            The second factor
        """
        if not A.qubits_out().match_nonzero(B.qubits_in()):
            raise ValueError(
                f"Non matching qubit maps {A.qubits_out()} and {B.qubits_in()}"
            )

        max_qubits = max(A.qubits_in().total_qubits,
                         B.qubits_out().total_qubits)
        qubits_in_A = A.qubits_in()
        self._qubits_in = QubitMap(
            qubits_in_A.registers,
            max_qubits - qubits_in_A.total_qubits,
        )
        qubits_out_B = B.qubits_out()
        self._qubits_out = QubitMap(
            qubits_out_B.registers,
            max_qubits - qubits_out_B.total_qubits,
        )
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit()
        circuit += self.B.circuit()

        return circuit

    def qubits_in(self) -> QubitMap:
        return self._qubits_in

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()

    def phase(self) -> float:
        return self.A.phase() + self.B.phase()

    def controlled(self) -> Node:
        A_controlled = self.A.controlled()
        B_controlled = self.B.controlled()
        return UnsafeMul(A_controlled, B_controlled)


class Tensor(Node):
    """
    Node representing the tensor product of two other nodes

    The order of operations is such that ``A`` corresponds to the lower
    significant digits of the index, i.e.

    >>> from bequem.nodes import Tensor, Identity, Increment
    >>> from bequem.qubit_map import QubitMap
    >>> import numpy as np
    >>> Tensor(Increment(1), Identity(QubitMap(1))).compute(np.eye(4))
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """
    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        """
        :param A:
            The first factor
        :param B:
            The second factor
        """
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape(
            batch_shape +
            [self.B.qubits_in().dimension,
             self.A.qubits_in().dimension])
        input = self.A.compute(input)
        input = np.swapaxes(input, -1, -2)
        input = self.B.compute(input)
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        batch_shape = list(input.shape[:-1])
        input = input.reshape(
            batch_shape +
            [self.B.qubits_out().dimension,
             self.A.qubits_out().dimension])
        input = self.A.compute_adjoint(input)
        input = np.swapaxes(input, -1, -2)
        input = self.B.compute_adjoint(input)
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def circuit(self) -> Circuit:
        qubits_in_A = self.A.qubits_in()
        qubits_in_B = self.B.qubits_in()

        circuit = Circuit()

        circuit_A = self.A.circuit().tq_circuit
        circuit.tq_circuit += circuit_A
        qubit_map_B = dict([(i, i + qubits_in_A.total_qubits)
                            for i in range(qubits_in_B.total_qubits)])
        circuit_B = self.B.circuit().tq_circuit.map_qubits(qubit_map_B)
        circuit.tq_circuit += circuit_B

        circuit.tq_circuit.n_qubits = qubits_in_A.total_qubits + qubits_in_B.total_qubits

        return circuit

    def qubits_in(self) -> QubitMap:
        qubits_A = self.A.qubits_in()
        qubits_B = self.B.qubits_in()
        return QubitMap(
            qubits_A.registers + qubits_B.registers,
        )

    def qubits_out(self) -> QubitMap:
        qubits_A = self.A.qubits_out()
        qubits_B = self.B.qubits_out()
        return QubitMap(
            qubits_A.registers + qubits_B.registers,
        )

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()

    def phase(self) -> float:
        return self.A.phase() + self.B.phase()


Node.__and__ = lambda A, B: Tensor(A, B)


class Adjoint(Node):
    """
    Node representing the adjoint of another node

    :ivar A:
        The node of which to compute the adjoint
    """
    A: Node

    def __init__(self, A: Node):
        """
        :param A:
            The node of which to compute the adjoint
        """
        self.A = A

    def children(self) -> list[Node]:
        return [self.A]

    def qubits_in(self) -> QubitMap:
        return self.A.qubits_out()

    def qubits_out(self) -> QubitMap:
        return self.A.qubits_in()

    def normalization(self) -> float:
        return self.A.normalization()

    def phase(self) -> float:
        return -self.A.phase()

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute_adjoint(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute(input)

    def circuit(self) -> Circuit:
        return self.A.circuit().adjoint()


class Scale(Node):
    """
    Node representing the product of a scalar and another node

    :ivar A:
        The node to scale
    :ivar scale:
        The scalar factor
    :ivar absolute:
        If ``True``, ``A`` is divided by its normalization first
    """
    A: Node
    scale: float
    absolute: bool
    remove_efficiency: bool

    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: float = 1,
        absolute: bool = False,
    ):
        """
        :param A:
            The node to scale
        :param scale:
            The scalar factor
        :param absolute:
            If ``True``, ``A`` is divided by its normalization first
        """
        self.A = A
        # TODO: assert_efficiency not implemented yet
        assert remove_efficiency == 1
        self.remove_efficiency = remove_efficiency
        self.scale = np.abs(scale)
        self.global_phase = np.angle(scale)
        self.absolute = absolute

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"scale": self.scale, "absolute": self.absolute}

    def qubits_in(self) -> QubitMap:
        return self.A.qubits_in()

    def qubits_out(self) -> QubitMap:
        return self.A.qubits_out()

    def normalization(self) -> float:
        if self.absolute:
            return self.scale
        else:
            return self.scale * self.A.normalization()

    def phase(self) -> float:
        if self.absolute:
            return np.angle(self.scale)
        else:
            return np.angle(self.scale) + self.A.phase()

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return self.scale / self.A.normalization() * self.A.compute(input)
        else:
            return self.scale * self.A.compute(input)

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return self.scale / self.A.normalization(
            ) * self.A.compute_adjoint(input)
        else:
            return self.scale * self.A.compute_adjoint(input)

    def circuit(self) -> Circuit:
        return self.A.circuit()


class ComputeProjection(Node):
    def __init__(self, qubits: QubitMap):
        self.qubits = QubitMap(qubits.registers, 1)

    def children(self) -> list[Node]:
        return []

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self.qubits

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def circuit(self) -> Circuit:
        # TODO
        return Circuit()

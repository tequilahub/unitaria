from __future__ import annotations
import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.subspace import Subspace
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
        if not A.subspace_out.match_nonzero(B.subspace_in):
            raise ValueError(
                f"Non matching qubit maps {A.subspace_out} and {B.subspace_in}"
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

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit
        circuit += self.B.circuit

        return circuit

    def _subspace_in(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits,
                         self.B.subspace_out.total_qubits)
        return Subspace(
            self.A.subspace_in.registers,
            max_qubits - self.A.subspace_in.total_qubits,
        )

    def _subspace_out(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits,
                         self.B.subspace_out.total_qubits)
        return Subspace(
            self.B.subspace_out.registers,
            max_qubits - self.B.subspace_out.total_qubits,
        )

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization

    def _phase(self) -> float:
        return self.A.phase + self.B.phase


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
        input = input.reshape([-1, self.A.subspace_in.dimension])
        input = self.A.compute(input)
        input = input.reshape(
            batch_shape +
            [self.B.subspace_in.dimension,
             self.A.subspace_out.dimension])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.subspace_in.dimension])
        input = self.B.compute(input)
        input = input.reshape(
            batch_shape +
            [self.A.subspace_out.dimension,
             self.B.subspace_out.dimension])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if input is None:
            input = np.array([1])
        batch_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.A.subspace_out.dimension])
        input = self.A.compute_adjoint(input)
        input = input.reshape(
            batch_shape +
            [self.B.subspace_out.dimension,
             self.A.subspace_in.dimension])
        input = np.swapaxes(input, -1, -2)
        input = input.reshape([-1, self.B.subspace_out.dimension])
        input = self.B.compute_adjoint(input)
        input = input.reshape(
            batch_shape +
            [self.A.subspace_in.dimension,
             self.B.subspace_in.dimension])
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def _circuit(self) -> Circuit:
        circuit = Circuit()

        circuit_A = self.A.circuit.tq_circuit
        circuit.tq_circuit += circuit_A
        qubit_map_B = dict([(i, i + self.A.subspace_in.total_qubits)
                            for i in range(self.B.subspace_in.total_qubits)])
        circuit_B = self.B.circuit.tq_circuit.map_qubits(qubit_map_B)
        circuit.tq_circuit += circuit_B

        circuit.tq_circuit.n_qubits = self.A.subspace_in.total_qubits + self.B.subspace_in.total_qubits

        return circuit

    def _subspace_in(self) -> Subspace:
        subspace_A = self.A.subspace_in
        subspace_B = self.B.subspace_in
        return Subspace(
            subspace_A.registers + subspace_B.registers,
        )

    def _subspace_out(self) -> Subspace:
        subspace_A = self.A.subspace_out
        subspace_B = self.B.subspace_out
        return Subspace(
            subspace_A.registers + subspace_B.registers,
        )

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization

    def _phase(self) -> float:
        return self.A.phase + self.B.phase


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

    def _subspace_in(self) -> Subspace:
        return self.A.subspace_out

    def _subspace_out(self) -> Subspace:
        return self.A.subspace_in

    def _normalization(self) -> float:
        return self.A.normalization

    def _phase(self) -> float:
        return -self.A.phase

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute_adjoint(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute(input)

    def _circuit(self) -> Circuit:
        return self.A.circuit.adjoint()


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

    def _subspace_in(self) -> Subspace:
        return self.A.subspace_in

    def _subspace_out(self) -> Subspace:
        return self.A.subspace_out

    def _normalization(self) -> float:
        if self.absolute:
            return self.scale
        else:
            return self.scale * self.A.normalization

    def _phase(self) -> float:
        if self.absolute:
            return np.angle(self.scale)
        else:
            return np.angle(self.scale) + self.A.phase

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return self.scale / self.A.normalization * self.A.compute(input)
        else:
            return self.scale * self.A.compute(input)

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return self.scale / self.A.normalization * self.A.compute_adjoint(input)
        else:
            return self.scale * self.A.compute_adjoint(input)

    def _circuit(self) -> Circuit:
        return self.A.circuit

Node.__rmul__ = lambda A, s: Scale(A, s)


class ComputeProjection(Node):
    def __init__(self, qubits: Subspace):
        self.qubits = Subspace(qubits.registers, 1)

    def children(self) -> list[Node]:
        return []

    def _subspace_in(self) -> Subspace:
        return self.qubits

    def _subspace_out(self) -> Subspace:
        return self.qubits

    def _normalization(self) -> float:
        return 1

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(self) -> Circuit:
        # TODO
        return Circuit()

import numpy as np
import tequila as tq

from .node import Node
from bequem.subspace import Subspace
from bequem.circuit import Circuit
from bequem.circuits.arithmetic import increment_circuit_single_ancilla, addition_circuit, const_addition_circuit
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.identity import Identity
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.nodes.permutation import PermuteRegisters
from bequem.nodes.basic_ops import Adjoint
from bequem.nodes.ring_ops import Mul


class Increment(Node):
    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError()
        self.bits = bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "bits": self.bits }

    def _subspace_in(self) -> Subspace:
        if self.bits <= 3:
            return Subspace(self.bits)
        else:
            return Subspace(self.bits, 1)

    def _subspace_out(self) -> Subspace:
        if self.bits <= 3:
            return Subspace(self.bits)
        else:
            return Subspace(self.bits, 1)

    def _normalization(self) -> float:
        return 1

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, 1, axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, -1, axis=-1)

    def _circuit(self) -> Circuit:
        if self.bits <= 3:
            circuit = Circuit()
            for i in reversed(range(self.bits)):
                circuit.tq_circuit += tq.gates.X(target=i, control=list(range(i)))
            return circuit
        else:
            circuit = increment_circuit_single_ancilla(target=list(reversed(range(self.bits))), ancilla=self.bits)
            return Circuit(circuit)


class IntegerAddition(Node):
    def __init__(self, source_bits: int, target_bits: int):
        # TODO: Restriction is because the ancilla free construction needs two source bits.
        #  I know how to fix this but haven't implemented it yet.
        if source_bits < 2 or target_bits < source_bits:
            raise ValueError()
        self.source_bits = source_bits
        self.target_bits = target_bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "source_bits": self.source_bits, "target_bits": self.target_bits }

    def _subspace_in(self) -> Subspace:
        return Subspace(self.source_bits + self.target_bits)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.source_bits + self.target_bits)

    def _normalization(self) -> float:
        return 1

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2 ** self.source_bits
        M = 2 ** self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], val, axis=-1)
        return result.reshape(old_shape)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2 ** self.source_bits
        M = 2 ** self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], -val, axis=-1)
        return result.reshape(old_shape)

    def _circuit(self) -> Circuit:
        source = list(reversed(range(self.source_bits)))
        target = list(reversed(range(self.source_bits, self.source_bits + self.target_bits)))
        circuit = addition_circuit(source, target)
        return Circuit(circuit)


class ConstantIntegerAddition(Node):
    bits: int
    constant: int

    def __init__(self, bits: int, constant: int):
        if bits < 1:
            raise ValueError()
        self.bits = bits
        self.constant = constant

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "bits": self.bits, "constant": self.constant }

    def _subspace_in(self) -> Subspace:
        return Subspace(self.bits, 2)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.bits, 2)

    def _normalization(self) -> float:
        return 1

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, self.constant, axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, -self.constant, axis=-1)

    def _circuit(self) -> Circuit:
        circuit = const_addition_circuit(list(reversed(range(self.bits))), self.constant, [self.bits, self.bits+1])
        circuit.n_qubits = self.subspace_in.total_qubits

        return Circuit(circuit)


class ConstantIntegerMultiplication(ProxyNode):
    bits: int
    constant: int

    def __init__(self, bits: int, constant: int):
        if constant & 1 != 1:
            raise ValueError(f"Constant factor {constant} is uneven. This would result in a non-reversible operation.")
        if bits < 1:
            raise ValueError()
        self.bits = bits
        self.constant = constant

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "bits": self.bits, "constant": self.constant }

    def definition(self) -> Node:
        if self.bits == 1:
            assert self.constant == 1
            return Identity(Subspace(1))
        result = None
        for i in reversed(range(self.bits - 1)):
            add_bits = self.bits - 1 - i
            c = (self.constant >> 1) & ((1 << add_bits) - 1)
            const_add = BlockDiagonal(Identity(Subspace(add_bits)), ConstantIntegerAddition(add_bits, c))
            permutation = PermuteRegisters(Subspace(add_bits + 1), [add_bits] + list(range(add_bits)))
            # TODO: The skip_projection can be removed onces this is done automatically
            const_add = Mul(Adjoint(permutation),
                            Mul(const_add, permutation, skip_projection=True),
                            skip_projection=True)
            const_add = Identity(Subspace(i)) & const_add
            if result is not None:
                result = Mul(result, const_add, skip_projection=True)
            else:
                result = const_add
        return result

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, 2 ** self.bits])
        indices = np.array([(i * self.constant) % 2 ** self.bits for i in range(2 ** self.bits)], dtype=np.uint32)
        output = input[:, indices]
        return output.reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        inv = self._mod_inv()
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, 2 ** self.bits])
        indices = np.array([(i * inv) % 2 ** self.bits for i in range(2 ** self.bits)], dtype=np.uint32)
        output = input[:, indices]
        return output.reshape(outer_shape + [-1])

    def _mod_inv(self):
        return pow(self.constant, -1, mod=2**self.bits)

from __future__ import annotations

import numpy as np
import tequila as tq
from bequem.nodes.node import Node
from bequem.nodes.proxy_node import ProxyNode
from bequem.qubit_map import QubitMap, ZeroQubit
from bequem.nodes.basic_ops import UnsafeMul, Adjoint
from bequem.circuit import Circuit


class Permutation(ProxyNode):

    qubits_from: QubitMap
    qubits_to: QubitMap

    def __init__(self, qubits_from: QubitMap, qubits_to: QubitMap):
        if qubits_from.dimension != qubits_to.dimension:
            raise ValueError
        self.qubits_from = qubits_from
        self.qubits_to = qubits_to

    def definition(self):
        perm_in = move_zeros_to_end(self.qubits_from)
        perm_out = move_zeros_to_end(self.qubits_to)
        if perm_in.qubits_out().match_nonzero(perm_out.qubits_out()):
            return UnsafeMul(perm_in, Adjoint(perm_out))
        raise NotImplementedError

    def parameters(self) -> dict:
        return { "qubits_from": self.qubits_from, "qubits_to": self.qubits_to }

    def qubits_in(self) -> QubitMap:
        max_qubits = max(self.qubits_from.total_qubits, self.qubits_to.total_qubits)
        return QubitMap(self.qubits_from.registers, max_qubits - self.qubits_from.total_qubits)

    def qubits_out(self) -> QubitMap:
        max_qubits = max(self.qubits_from.total_qubits, self.qubits_to.total_qubits)
        return QubitMap(self.qubits_to.registers, max_qubits - self.qubits_to.total_qubits)

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input


def move_zeros_to_end(qubits: QubitMap) -> PermuteRegisters:
    nonzero_registers = []
    zero_registers = []
    for (i, register) in enumerate(qubits.registers):
        if isinstance(register, ZeroQubit):
            zero_registers.append(i)
        else:
            nonzero_registers.append(i)

    return PermuteRegisters(qubits, nonzero_registers + zero_registers)


class PermuteRegisters(Node):
    """
    Operation permuting the registers of a state
    
    Permutes a vector in the space defined by ``qubits``
    such that the i-th register after the operation will be
    ``qubits.registers[permutation_map[i]]``.
    """

    qubits: QubitMap
    permutation_map: list[int]

    def __init__(self, qubits: QubitMap, permutation_map: list[int]):
        self.qubits = qubits
        self.permutation_map = permutation_map

    def parameters(self) -> dict:
        return { "qubits": self.qubits, "permutation_map": self.permutation_map }

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return QubitMap([self.qubits.registers[i] for i in self.permutation_map])

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        register_shape = [r.dimension() for r in self.qubits.registers]
        total_shape = outer_shape + register_shape
        input = input.reshape(outer_shape + register_shape)
        perm = list(range(len(outer_shape))) + [len(total_shape) - i - 1 for i in reversed(self.permutation_map)]
        input = np.transpose(input, perm)
        return input.reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        register_shape = [r.dimension() for r in self.qubits_out().registers]
        total_shape = outer_shape + register_shape
        input = input.reshape(outer_shape + register_shape)
        perm = list(range(len(outer_shape))) + [len(total_shape) - i - 1 for i in reversed(self.permutation_map)]
        input = np.transpose(input, np.argsort(perm))
        return input.reshape(outer_shape + [-1])

    def circuit(self) -> Circuit:

        if self.qubits.total_qubits == 0:
            circuit = Circuit()
            circuit.tq_circuit.n_qubits = 1
            return circuit

        register_qubits = []
        qubit_index = 0
        for register in self.qubits.registers:
            register_qubits.append(list(range(qubit_index, qubit_index + register.total_qubits())))
            qubit_index += register.total_qubits()

        permutation_map_qubits = sum([register_qubits[i] for i in self.permutation_map], [])

        circuit = Circuit()

        i = 0
        n = 0
        while i < self.qubits.total_qubits:
            j = permutation_map_qubits[i]
            if i != j:
                circuit.tq_circuit += tq.gates.SWAP(i, j)
                permutation_map_qubits[i], permutation_map_qubits[j] = permutation_map_qubits[j], permutation_map_qubits[i]
                n += 1
                if n > self.qubits.total_qubits:
                    raise ValueError(f"{self.permutation_map} is not a permutation")
            else:
                i += 1

        circuit.tq_circuit = circuit.tq_circuit.dagger()

        circuit.tq_circuit.n_qubits = self.qubits.total_qubits

        return circuit
    


def _find_matching_partitioning(
    a: QubitMap, b: QubitMap
) -> list[tuple[QubitMap, QubitMap]]:
    """
    Finds a partitoning of a and b, such that the ith subdivision of either
    partitioning has the same dimension.

    Neither QubitMap should contian ZeroQubits in its register.
    """
    if a.registers == [] and b.registers == []:
        return []

    assert a.dimension == b.dimension
    subdivisions = []
    last_breakpoint_a = 0
    last_breakpoint_b = 0
    i_a = 1
    i_b = 1
    submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
    submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
    while i_a < len(a.registers) and i_b < len(b.registers):
        if submap_a.dimension == submap_b.dimension:
            subdivisions.append((submap_a, submap_b))
            last_breakpoint_a = i_a
            last_breakpoint_b = i_b
            i_a += 1
            i_b += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
        elif submap_a.dimension < submap_b.dimension:
            i_a += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
        else:
            i_b += 1
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])

    submap_a = QubitMap(a.registers[last_breakpoint_a:])
    submap_b = QubitMap(b.registers[last_breakpoint_b:])
    subdivisions.append((submap_a, submap_b))

    return subdivisions

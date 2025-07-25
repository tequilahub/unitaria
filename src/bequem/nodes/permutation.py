from __future__ import annotations

import numpy as np
import tequila as tq
from bequem.nodes.node import Node
from bequem.nodes.proxy_node import ProxyNode
from bequem.subspace import Subspace, ZeroQubit
from bequem.nodes.basic_operations.unsafe_multiplication import UnsafeMul
from bequem.nodes.basic_operations.adjoint import Adjoint
from bequem.circuit import Circuit


class Permutation(ProxyNode):
    subspace_from: Subspace
    subspace_to: Subspace

    def __init__(self, subspace_from: Subspace, subspace_to: Subspace):
        if subspace_from.dimension != subspace_to.dimension:
            raise ValueError(f"dimensions {subspace_from.dimension} and {subspace_to.dimension} do not match")
        self.subspace_from = subspace_from
        self.subspace_to = subspace_to

    def definition(self):
        perm_in = move_zeros_to_end(self.subspace_from)
        perm_out = move_zeros_to_end(self.subspace_to)
        if perm_in.subspace_out.match_nonzero(perm_out.subspace_out):
            return UnsafeMul(perm_in, Adjoint(perm_out))
        raise NotImplementedError

    def parameters(self) -> dict:
        return {"subspace_from": self.subspace_from, "subspace_to": self.subspace_to}

    def _subspace_in(self) -> Subspace:
        max_qubits = max(self.subspace_from.total_qubits, self.subspace_to.total_qubits)
        return Subspace(self.subspace_from.registers, max_qubits - self.subspace_from.total_qubits)

    def _subspace_out(self) -> Subspace:
        max_qubits = max(self.subspace_from.total_qubits, self.subspace_to.total_qubits)
        return Subspace(self.subspace_to.registers, max_qubits - self.subspace_to.total_qubits)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input


def move_zeros_to_end(qubits: Subspace) -> PermuteRegisters:
    nonzero_registers = []
    zero_registers = []
    for i, register in enumerate(qubits.registers):
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

    qubits: Subspace
    permutation_map: list[int]

    def __init__(self, qubits: Subspace, permutation_map: list[int]):
        self.qubits = qubits
        self.permutation_map = permutation_map

    def parameters(self) -> dict:
        return {"qubits": self.qubits, "permutation_map": self.permutation_map}

    def _subspace_in(self) -> Subspace:
        return self.qubits

    def _subspace_out(self) -> Subspace:
        return Subspace([self.qubits.registers[i] for i in self.permutation_map])

    def _normalization(self) -> float:
        return 1

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
        register_shape = [r.dimension() for r in self.subspace_out.registers]
        total_shape = outer_shape + register_shape
        input = input.reshape(outer_shape + register_shape)
        perm = list(range(len(outer_shape))) + [len(total_shape) - i - 1 for i in reversed(self.permutation_map)]
        input = np.transpose(input, np.argsort(perm))
        return input.reshape(outer_shape + [-1])

    def _circuit(self) -> Circuit:
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
                permutation_map_qubits[i], permutation_map_qubits[j] = (
                    permutation_map_qubits[j],
                    permutation_map_qubits[i],
                )
                n += 1
                if n > self.qubits.total_qubits:
                    raise ValueError(f"{self.permutation_map} is not a permutation")
            else:
                i += 1

        circuit.tq_circuit = circuit.tq_circuit.dagger()

        circuit.tq_circuit.n_qubits = self.qubits.total_qubits

        return circuit


def _find_matching_partitioning(a: Subspace, b: Subspace) -> list[tuple[Subspace, Subspace]]:
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
    submap_a = Subspace(a.registers[last_breakpoint_a:i_a])
    submap_b = Subspace(b.registers[last_breakpoint_b:i_b])
    while i_a < len(a.registers) and i_b < len(b.registers):
        if submap_a.dimension == submap_b.dimension:
            subdivisions.append((submap_a, submap_b))
            last_breakpoint_a = i_a
            last_breakpoint_b = i_b
            i_a += 1
            i_b += 1
            submap_a = Subspace(a.registers[last_breakpoint_a:i_a])
            submap_b = Subspace(b.registers[last_breakpoint_b:i_b])
        elif submap_a.dimension < submap_b.dimension:
            i_a += 1
            submap_a = Subspace(a.registers[last_breakpoint_a:i_a])
        else:
            i_b += 1
            submap_b = Subspace(b.registers[last_breakpoint_b:i_b])

    submap_a = Subspace(a.registers[last_breakpoint_a:])
    submap_b = Subspace(b.registers[last_breakpoint_b:])
    subdivisions.append((submap_a, submap_b))

    return subdivisions

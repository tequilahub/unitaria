from __future__ import annotations

import numpy as np
from bequem.nodes.proxy_node import ProxyNode
from bequem.qubit_map import QubitMap
from bequem.nodes.identity import Identity


class Permutation(ProxyNode):

    qubits_from: QubitMap
    qubits_to: QubitMap

    def __init__(self, qubits_from: QubitMap, qubits_to: QubitMap):
        if qubits_from.dimension != qubits_to.dimension:
            raise ValueError
        self.qubits_from = qubits_from
        self.qubits_to = qubits_to

    def definition(self):
        if self.qubits_in() == self.qubits_out():
            return Identity(self.qubits_in())
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


def _find_matching_subdivision(
    a: QubitMap, b: QubitMap
) -> list[tuple[QubitMap, QubitMap]]:
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

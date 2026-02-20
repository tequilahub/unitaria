from __future__ import annotations

from typing import Sequence

import numpy as np
import tequila as tq
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace, ZeroQubit, ControlledSubspace
from unitaria.nodes.basic.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.identity import Identity
from unitaria.circuit import Circuit
from unitaria.util import logreduce


def _move_zeros_to_end(subspace: Subspace) -> PermuteRegisters:
    nonzero_registers = []
    zero_registers = []
    for i, register in enumerate(subspace.registers):
        if isinstance(register, ZeroQubit):
            zero_registers.append(i)
        else:
            nonzero_registers.append(i)

    return PermuteRegisters(subspace, nonzero_registers + zero_registers)


class PermuteRegisters(Node):
    """
    Operation permuting the registers of a state

    Permutes a vector in the space defined by ``qubits``
    such that the i-th register after the operation will be
    ``qubits.registers[permutation_map[i]]``.
    """

    subspace: Subspace
    permutation_map: list[int]

    def __init__(self, subspace: Subspace, permutation_map: list[int]):
        super().__init__(subspace.dimension, subspace.dimension)
        self.subspace = subspace
        self.permutation_map = permutation_map

    def parameters(self) -> dict:
        return {"subspace": self.subspace, "permutation_map": self.permutation_map}

    def _subspace_in(self) -> Subspace:
        return self.subspace

    def _subspace_out(self) -> Subspace:
        return Subspace([self.subspace.registers[i] for i in self.permutation_map])

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        register_shape = [r.dimension() for r in reversed(self.subspace.registers)]
        total_shape = outer_shape + register_shape
        input = input.reshape(total_shape)
        perm = list(range(len(outer_shape))) + [len(total_shape) - i - 1 for i in reversed(self.permutation_map)]
        input = np.transpose(input, perm)
        return input.reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        register_shape = [r.dimension() for r in reversed(self.subspace_out.registers)]
        total_shape = outer_shape + register_shape
        input = input.reshape(total_shape)
        perm = list(range(len(outer_shape))) + [
            len(total_shape) - i - 1 for i in reversed(np.argsort(self.permutation_map))
        ]
        input = np.transpose(input, perm)
        return input.reshape(outer_shape + [-1])

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        if self.subspace.total_qubits == 0:
            return Circuit()

        register_qubits = []
        qubit_index = 0
        for register in self.subspace.registers:
            register_qubits.append(list(range(qubit_index, qubit_index + register.total_qubits())))
            qubit_index += register.total_qubits()

        permutation_map_qubits = sum([register_qubits[i] for i in self.permutation_map], [])

        circuit = Circuit()

        i = 0
        n = 0
        while i < self.subspace.total_qubits:
            j = permutation_map_qubits[i]
            if i != j:
                circuit += tq.gates.SWAP(target[i], target[j])
                permutation_map_qubits[i], permutation_map_qubits[j] = (
                    permutation_map_qubits[j],
                    permutation_map_qubits[i],
                )
                n += 1
                if n > self.subspace.total_qubits:
                    raise ValueError(f"{self.permutation_map} is not a permutation")
            else:
                i += 1

        return circuit.adjoint()

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0


def _controlled_trailing_zeros(subspace: Subspace):
    case_zero = subspace.case_zero()
    case_one = subspace.case_one()
    return min(case_zero.trailing_zeros(), case_one.trailing_zeros())


class AddZerosToControl(Node):
    """
    Operation to add zeros to both branches of a controlled subspace.

    Specifically, this implements the identity for ``subspace_in = subspace & z`` and ``subspace_out = (subspace.case_zero() & z) | (subspace.case_one() & z)`` where ``z = Subspace(0, num_zeros)``
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace, num_zeros: int):
        assert len(subspace.registers) > 0 and isinstance(subspace.registers[-1], ControlledSubspace)
        super().__init__(subspace.dimension, subspace.dimension)
        self.subspace = subspace
        self.num_zeros = num_zeros

    def remove_zeros(subspace: Subspace, num_zeros: int):
        assert num_zeros > 0
        return Adjoint(
            AddZerosToControl(
                Subspace(subspace.case_zero().registers[:-num_zeros])
                | Subspace(subspace.case_one().registers[:-num_zeros]),
                num_zeros,
            )
        )

    def parameters(self) -> dict:
        return {"subspace": self.subspace, "num_zeros": self.num_zeros}

    def _subspace_in(self) -> Subspace:
        return self.subspace & Subspace(0, self.num_zeros)

    def _subspace_out(self) -> Subspace:
        zeros = Subspace(0, self.num_zeros)
        return (self.subspace.case_zero() & zeros) | (self.subspace.case_one() & zeros)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()

        for i in reversed(range(self.num_zeros)):
            circuit += tq.gates.SWAP(target[-(i + 1)], target[-(i + 2)])

        return circuit

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0


class SubspaceRightRotation(Node):
    """
    Identity operation rotating the root bit of the subspace.

    Takes subspace of the form ``(l | m) | (r & Subspace(0, 1))`` to ``(l & Subspace(0, 1)) | (m | r)``.
    """

    def __init__(self, subspace: Subspace):
        assert len(subspace.registers) > 0 and isinstance(subspace.registers[-1], ControlledSubspace)
        super().__init__(subspace.dimension, subspace.dimension)

        pivot = subspace.case_zero()
        assert len(pivot.registers) > 0 and isinstance(pivot.registers[-1], ControlledSubspace)
        L = pivot.case_zero()
        M = pivot.case_one()
        R = subspace.case_one()
        assert R.trailing_zeros() >= 1
        self.subspace_out = Subspace(
            [
                ControlledSubspace(
                    Subspace(L.registers, 1), Subspace([ControlledSubspace(M, Subspace(R.registers[:-1]))])
                )
            ],
        )

        self.subspace = subspace

    def left_rotate(subspace: Subspace) -> Node:
        pivot = subspace.case_one()
        assert len(pivot.registers) > 0 and isinstance(pivot.registers[-1], ControlledSubspace)
        L = subspace.case_zero()
        M = pivot.case_zero()
        R = pivot.case_one()
        assert L.trailing_zeros() >= 1
        subspace_in = Subspace(
            [
                ControlledSubspace(
                    Subspace([ControlledSubspace(Subspace(L.registers[:-1]), M)]),
                    Subspace(R.registers, 1),
                )
            ],
        )
        return Adjoint(SubspaceRightRotation(subspace_in))

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"subspace": self.subspace}

    def _subspace_in(self) -> Subspace:
        return self.subspace

    def _subspace_out(self) -> Subspace:
        return self.subspace_out

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(self, target, clean_ancillae, borrowed_ancillae) -> Circuit:
        circuit = Circuit()
        circuit += tq.gates.CNOT(target[-2], target[-1])
        circuit += tq.gates.CNOT(target[-1], target[-2])

        circuit.n_qubits = self.subspace_out.total_qubits

        return circuit

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0


def _rotate(subspace: Subspace, right: bool) -> Node:
    """
    Performs a rotation on the root of the given subspace.

    This method works with all subspaces. The resulting node has
    ``node.subspace_in.matches_nonzero(subspace)``.
    """
    permutation = []

    if right:
        other = subspace.case_one()
    else:
        other = subspace.case_zero()

    trailing_zeros = _controlled_trailing_zeros(subspace)
    other_zeros = other.trailing_zeros()
    if other_zeros > 1 and trailing_zeros > 0:
        permutation.append(
            AddZerosToControl.remove_zeros(Subspace(subspace.nonzero_registers()), min(trailing_zeros, other_zeros - 1))
        )
        subspace = permutation[-1].subspace_out
    elif other_zeros == 0:
        permutation.append(AddZerosToControl(Subspace(subspace.nonzero_registers()), 1))
        subspace = permutation[-1].subspace_out

    if right:
        pivot = subspace.case_zero()
    else:
        pivot = subspace.case_one()

    pivot_zeros = pivot.trailing_zeros()
    if pivot_zeros != 0:
        move = AddZerosToControl(Subspace(pivot.nonzero_registers()), pivot_zeros)
        if right:
            permutation.append(move | Identity(subspace.case_one()))
        else:
            permutation.append(Identity(subspace.case_zero()) | move)
        subspace = permutation[-1].subspace_out

    if right:
        permutation.append(SubspaceRightRotation(Subspace(subspace.nonzero_registers())))
    else:
        permutation.append(SubspaceRightRotation.left_rotate(Subspace(subspace.nonzero_registers())))
    return logreduce(UnsafeMul, permutation)


def _rotate_to(subspace: Subspace, index: int) -> Node:
    assert index >= 1 and index <= subspace.dimension - 1

    permutation = [Identity(subspace)]

    while subspace.case_zero().dimension != index:
        if subspace.case_zero().dimension > index:
            pivot = subspace.case_zero()
            if pivot.dimension > 1 and pivot.case_zero().dimension < index:
                rotate = _rotate(pivot, False)
                added_zeros = rotate.subspace_in.total_qubits - pivot.total_qubits
                rotate = rotate | Identity(subspace.case_one())
                # Now rotate.subspace_in.match_nonzero(subspace) might not hold,
                # since _rotate can introduce additional zero bits that cannot
                # be declared as ancillas. We move these zero bits to the end.
                if added_zeros > 0:
                    permutation.append(AddZerosToControl(subspace, added_zeros))
                permutation.append(rotate)
            permutation.append(_rotate(permutation[-1].subspace_out, True))

            assert permutation[-1].subspace_out.case_zero().dimension < subspace.case_zero().dimension, (
                "Not making progress. This is a bug."
            )
        else:
            pivot = subspace.case_one()
            if pivot.dimension > 1 and pivot.case_zero().dimension > index:
                rotate = _rotate(pivot, True)
                added_zeros = rotate.subspace_in.total_qubits - pivot.total_qubits
                rotate = Identity(subspace.case_zero()) | rotate
                # Now rotate.subspace_in.match_nonzero(subspace) might not hold,
                # since _rotate can introduce additional zero bits that cannot
                # be declared as ancillas. We move these zero bits to the end.
                if added_zeros > 0:
                    permutation.append(AddZerosToControl(subspace, added_zeros))
                permutation.append(rotate)
            permutation.append(_rotate(permutation[-1].subspace_out, False))

            assert permutation[-1].subspace_out.case_zero().dimension > subspace.case_zero().dimension, (
                "Not making progress. This is a bug."
            )

        subspace = permutation[-1].subspace_out

    if len(permutation) == 1:
        # Contains just the identity
        return permutation[0]
    else:
        return logreduce(UnsafeMul, permutation[1:])


def _indices_from_root_to(a: Subspace, index: int) -> int:
    if a.dimension == 1:
        return [0]
    root_index = a.case_zero().dimension
    if root_index == index:
        return [index]
    if root_index > index:
        return [root_index] + _indices_from_root_to(a.case_zero(), index)
    if root_index < index:
        return [root_index] + [i + root_index for i in _indices_from_root_to(a.case_one(), index - root_index)]


def _find_good_pivot_index(a: Subspace, b: Subspace) -> int:
    half = a.dimension // 2

    indices_a = _indices_from_root_to(a, half)
    assert len(indices_a) > 0
    indices_b = _indices_from_root_to(b, half)
    assert len(indices_b) > 0

    product = [(ia, na + nb) for na, ia in enumerate(indices_a) for nb, ib in enumerate(indices_b) if ia == ib]

    return min(product, key=lambda x: x[1])[0]


def permute(a: Subspace, b: Subspace) -> tuple[Node, Node]:
    """
    Implements the necessary permutation between two subspaces.

    In matrix arithmetic form this is just an identity operation, however the
    mapping to qubits in input and output may be different. Both subspaces have
    to have the same dimension.

    :param a:
        The source subspace
    :param b:
        The target subspace

    The results will be two nodes ``perm_a, perm_b`` implementing identity and
    such that

    * ``perm_a.subspace_in.matches_nonzero(a)``
    * ``perm_b.subspace_in.matches_nonzero(b)``
    * ``perm_b.subspace_out.matches_nonzero(perm_a.subspace_out)``

    This latter subspace will often be more balanced than the other two.
    """
    if a.dimension != b.dimension:
        raise ValueError(f"dimensions {a.dimension} and {b.dimension} do not match")

    if a.match_nonzero(b):
        return Identity(a), Identity(b)
    perm_a = _move_zeros_to_end(a)
    perm_b = _move_zeros_to_end(b)
    a = perm_a.subspace_out
    b = perm_b.subspace_out

    partitioning = _find_matching_partitioning(a, b)

    subperms_a, subperms_b = zip(*[_brute_force(a, b) for a, b in partitioning])

    subperms_a = logreduce(lambda x, y: x & y, subperms_a)
    move_a_in = _move_zeros_to_end(subperms_a.subspace_in)
    move_a_out = _move_zeros_to_end(subperms_a.subspace_out)
    perm_a = UnsafeMul(UnsafeMul(perm_a, Adjoint(move_a_in)), UnsafeMul(subperms_a, move_a_out))
    subperms_b = logreduce(lambda x, y: x & y, subperms_b)
    move_b_in = _move_zeros_to_end(subperms_b.subspace_in)
    move_b_out = _move_zeros_to_end(subperms_b.subspace_out)
    perm_b = UnsafeMul(UnsafeMul(perm_b, Adjoint(move_b_in)), UnsafeMul(subperms_b, move_b_out))

    return perm_a, perm_b


def _brute_force(a: Subspace, b: Subspace) -> tuple[Node, Node]:
    assert a.dimension == b.dimension

    if a.match_nonzero(b):
        return Identity(a), Identity(b)

    perm_a = None
    perm_b = None

    if a.case_zero().dimension != b.case_zero().dimension:
        index = _find_good_pivot_index(a, b)

        perm_a = _rotate_to(a, index)
        perm_b = _rotate_to(b, index)

        a = perm_a.subspace_out
        b = perm_b.subspace_out

    a0 = a.case_zero()
    b0 = b.case_zero()
    assert a0.dimension == b0.dimension

    a1 = a.case_one()
    b1 = b.case_one()

    assert a0.dimension < a.dimension
    assert a1.dimension < a.dimension

    perm_a0, perm_b0 = permute(a0, b0)
    perm_a1, perm_b1 = permute(a1, b1)

    perm_a01 = perm_a0 | perm_a1
    perm_b01 = perm_b0 | perm_b1

    # perm_a0, ..., perm_b1 may contain extra zero qubits. The | operator
    # equalizes the zero qubits between perm_*0 and perm_*1. The remaining
    # additional zero qubits have to be moved out of the control.
    added_zeros_a = perm_a01.subspace_in.total_qubits - a0.total_qubits - 1
    if added_zeros_a:
        move = AddZerosToControl.remove_zeros(perm_a01, added_zeros_a)
        perm_a01 = UnsafeMul(Adjoint(move), UnsafeMul(perm_a01, move))
    added_zeros_b = perm_b01.subspace_in.total_qubits - b0.total_qubits - 1
    if added_zeros_b:
        move = AddZerosToControl.remove_zeros(perm_b01, added_zeros_b)
        perm_b01 = UnsafeMul(Adjoint(move), UnsafeMul(perm_b01, move))

    if perm_a is not None:
        perm_a = UnsafeMul(perm_a, perm_a01)
        perm_b = UnsafeMul(perm_b, perm_b01)
    else:
        perm_a = perm_a01
        perm_b = perm_b01

    return perm_a, perm_b


def _find_matching_partitioning(a: Subspace, b: Subspace) -> list[tuple[Subspace, Subspace]]:
    """
    Finds a partitoning of a and b, such that the ith subdivision of either
    partitioning has the same dimension.

    Neither Subspace should contian ZeroQubits in its register.
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
